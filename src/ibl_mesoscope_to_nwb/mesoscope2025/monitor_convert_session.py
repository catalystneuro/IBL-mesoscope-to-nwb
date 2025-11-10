import time
import psutil
from threading import Thread


def format_hms(seconds: float) -> str:
    """Return 'Xh Ym Zs' style string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"


def monitor_usage(process, cpu_log, mem_log, flag):
    while not flag["stop"]:
        cpu_log.append(process.cpu_percent(interval=1))  # % of one core
        mem_log.append(process.memory_info().rss)  # bytes


def run_with_monitor(fn, *args, **kwargs):
    p = psutil.Process()
    cpu_log, mem_log = [], []
    flag = {"stop": False}

    t = Thread(target=monitor_usage, args=(p, cpu_log, mem_log, flag))
    t.start()

    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()

    flag["stop"] = True
    t.join()

    elapsed_sec = t1 - t0
    return out, elapsed_sec, cpu_log, mem_log


from ibl_mesoscope_to_nwb.mesoscope2025.convert_session import session_to_nwb
from pathlib import Path

data_dir_path = Path(r"E:\IBL-data-share\cortexlab\Subjects\SP061\2025-01-28\001")
output_dir_path = Path(r"E:\ibl_mesoscope_conversion_nwb")
eid = "5ce2e17e-8471-42d4-8a16-21949710b328"
subject_id = "SP061"
stub_test = False
overwrite = True

result, elapsed_sec, cpu_samples, mem_samples = run_with_monitor(
    # session_to_nwb function here
    session_to_nwb,
    # args for session_to_nwb
    data_dir_path,
    output_dir_path,
    eid,
    subject_id,
    stub_test,
    overwrite,
)

mean_cpu_percent = sum(cpu_samples) / len(cpu_samples)
ncores = psutil.cpu_count(logical=True)
mean_fraction = (mean_cpu_percent / 100) / ncores
print(f"Elapsed: {format_hms(elapsed_sec)}")
print(f"CPU % average: {sum(cpu_samples) / len(cpu_samples):.1f}%")
print(f"Mean CPU load fraction = {mean_fraction:.2f}")
print(f"Peak MEM: {max(mem_samples) / 1024 / 1024:.1f} MB")
