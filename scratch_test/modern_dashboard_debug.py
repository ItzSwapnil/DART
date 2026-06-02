import os
import signal
import sys
import traceback


# Register SIGUSR1 handler to dump stacks of all threads
def dump_stacks(sig, frame):
    print("--- DUMPING STACKS ---", file=sys.stderr, flush=True)
    for thread_id, stack in sys._current_frames().items():
        print(f"\nThread {thread_id}:", file=sys.stderr, flush=True)
        traceback.print_stack(stack, file=sys.stderr)
    print("--- END DUMPING ---", file=sys.stderr, flush=True)


signal.signal(signal.SIGUSR1, dump_stacks)

# Now read and execute the original dashboard code
dashboard_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui", "modern_dashboard.py")
with open(dashboard_path, encoding="utf-8") as f:
    code = f.read()

exec(code, globals())
