//! The threads the passes that use every core run on (`std`).
//!
//! A build's sort, a merge, a directory fill and a load's walk over the
//! runs all spread their work the same way: the work is cut into tasks,
//! [`run_tasks`] hands them out to up to [`threads`] threads, and the
//! calling thread takes tasks like every other one. [`threads`] is the
//! machine's hardware parallelism, capped by [`set_max_threads`]: a server
//! that serves searches from the same cores a background compaction runs on
//! can leave the compaction a core or two, or keep it on the calling thread
//! altogether.
//!
//! The cap is process-wide and read once per pass. What a pass computes
//! does not depend on it: the tasks of a sort, a merge or a walk cover the
//! same work whichever thread runs them.

use alloc::vec::Vec;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::thread;

/// The cap [`set_max_threads`] set, or 0 for none.
static MAX_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Cap the threads a pass that uses every core may take, at `n`; 0
/// removes the cap.
///
/// The setting is process-wide, and it applies from the next pass on:
/// `set_max_threads(1)` keeps every sort, merge, directory fill and load
/// on the calling thread, which is how a server can compact in the
/// background without taking the cores its searches run on. A search
/// takes the threads its `SearchOptions::threads` asks for, at most the
/// hardware's, so the cap does not change it. Results never depend on the
/// cap.
///
/// [`max_threads`] reports what a pass would take now.
pub fn set_max_threads(n: usize) {
    // Relaxed is enough: a thread that reads the cap is started after the
    // call (or after something that synchronizes with it), and a pass only
    // reads one number.
    MAX_THREADS.store(n, Ordering::Relaxed);
}

/// The number of threads a pass that uses every core would run on now: the
/// hardware parallelism, capped by [`set_max_threads`].
#[must_use]
pub fn max_threads() -> usize {
    threads()
}

/// Hardware threads, capped, or 1 if the platform does not say.
pub(crate) fn threads() -> usize {
    let hardware = hardware();
    match MAX_THREADS.load(Ordering::Relaxed) {
        0 => hardware,
        cap => hardware.min(cap),
    }
}

/// The hardware parallelism, asked once per process. On Linux the answer
/// takes dozens of system calls (the affinity mask and the cgroup's CPU
/// quota), tens of microseconds that a small merge of `Catalog::add`
/// would otherwise pay every time.
pub(crate) fn hardware() -> usize {
    static HARDWARE: OnceLock<usize> = OnceLock::new();
    *HARDWARE.get_or_init(|| thread::available_parallelism().map_or(1, NonZeroUsize::get))
}

/// Run `task(i)` for every `i` in `0..tasks` on up to `threads` threads,
/// each taking the next task until none are left. The calling thread
/// works too, so a thread that cannot be spawned costs nothing but
/// parallelism.
pub(crate) fn run_tasks(threads: usize, tasks: usize, task: impl Fn(usize) + Sync) {
    let next = AtomicUsize::new(0);
    let worker = || {
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= tasks {
                break;
            }
            task(i);
        }
    };
    if threads <= 1 || tasks <= 1 {
        worker();
        return;
    }
    thread::scope(|s| {
        for _ in 1..threads.min(tasks) {
            // A failed spawn leaves its share to the others.
            // The closure captures only references, so it is `Copy`.
            let _ = thread::Builder::new().spawn_scoped(s, worker);
        }
        worker();
    });
}

/// A slot per task for what it hands back.
pub(crate) fn slots<T>(n: usize) -> Vec<Mutex<Option<T>>> {
    (0..n).map(|_| Mutex::new(None)).collect()
}

/// A slot's lock, ignoring poisoning: a task that panicked has already
/// stopped the pass.
pub(crate) fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cap is what a pass takes, and 0 is no cap at all. This is the
    /// only test that writes the process-wide cap, and it leaves it off;
    /// the other one below names its thread count, so it never reads the
    /// cap and they can run at the same time.
    #[test]
    fn the_cap_bounds_the_threads_a_pass_takes() {
        let hardware = threads();
        assert!(hardware >= 1);
        set_max_threads(1);
        assert_eq!(threads(), 1);
        assert_eq!(max_threads(), 1);
        set_max_threads(2);
        assert_eq!(threads(), hardware.min(2));
        assert_eq!(max_threads(), hardware.min(2));
        set_max_threads(0);
        assert_eq!(threads(), hardware);
        assert_eq!(max_threads(), hardware);
    }

    /// Every task runs once, whatever the thread count, and a count of one
    /// runs them on the calling thread in the order they were handed out.
    #[test]
    fn every_task_runs_once() {
        for threads in [1usize, 2, 7] {
            for tasks in [0usize, 1, 5, 64] {
                let hits: Vec<AtomicUsize> = (0..tasks).map(|_| AtomicUsize::new(0)).collect();
                run_tasks(threads, tasks, |i| {
                    hits[i].fetch_add(1, Ordering::Relaxed);
                });
                for (i, hit) in hits.iter().enumerate() {
                    assert_eq!(
                        hit.load(Ordering::Relaxed),
                        1,
                        "{threads} threads, task {i}"
                    );
                }
            }
        }
        let order = Mutex::new(Vec::new());
        run_tasks(1, 8, |i| lock(&order).push(i));
        assert_eq!(
            order.into_inner().unwrap_or_else(PoisonError::into_inner),
            (0..8).collect::<Vec<_>>()
        );
    }
}
