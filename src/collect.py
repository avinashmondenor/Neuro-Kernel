import psutil
import pandas as pd
import time
from datetime import datetime

data = []
previous = {}

print("======================================")
print("      NeuroKernel Data Collector")
print("======================================")
print("Collecting process telemetry...")
print("Press Ctrl+C to stop.")

try:
    while True:

        current_time = time.time()
        timestamp = datetime.now()

        for p in psutil.process_iter([
            "pid",
            "name",
            "status",
            "ppid",
            "memory_info",
            "memory_percent",
            "num_threads",
            "cpu_times",
            "io_counters",
            "num_ctx_switches"
        ]):

            try:

                info = p.info
                pid = info["pid"]

                cpu_total = (
                    info["cpu_times"].user +
                    info["cpu_times"].system
                )

                read_bytes = (
                    info["io_counters"].read_bytes
                    if info["io_counters"] else 0
                )

                write_bytes = (
                    info["io_counters"].write_bytes
                    if info["io_counters"] else 0
                )

                cpu_percent = 0
                cpu_delta = 0
                read_rate = 0
                write_rate = 0

                if pid in previous:

                    old = previous[pid]

                    elapsed = current_time - old["time"]

                    if elapsed > 0:

                        cpu_delta = (
                            cpu_total - old["cpu"]
                        )

                        read_delta = (
                            read_bytes - old["read"]
                        )

                        write_delta = (
                            write_bytes - old["write"]
                        )

                        if cpu_delta >= 0:
                            cpu_percent = (
                                cpu_delta / elapsed
                            ) * 100

                        if read_delta >= 0:
                            read_rate = (
                                read_delta / elapsed
                            )

                        if write_delta >= 0:
                            write_rate = (
                                write_delta / elapsed
                            )

                previous[pid] = {
                    "time": current_time,
                    "cpu": cpu_total,
                    "read": read_bytes,
                    "write": write_bytes
                }

                data.append({

                    "timestamp": timestamp,

                    "pid": pid,

                    "process_name":
                        info["name"],

                    "process_state":
                        info["status"],

                    "parent_pid":
                        info["ppid"],

                    "cpu_percent":
                        round(max(0, cpu_percent), 2),

                    "memory_mb":
                        round(
                            info["memory_info"].rss /
                            (1024 * 1024),
                            2
                        ),

                    "memory_percent":
                        round(
                            info["memory_percent"],
                            3
                        ),

                    "threads":
                        info["num_threads"],

                    "context_switches":
                        (
                            info["num_ctx_switches"].voluntary +
                            info["num_ctx_switches"].involuntary
                        ),

                    "cpu_user_time":
                        info["cpu_times"].user,

                    "cpu_system_time":
                        info["cpu_times"].system,

                    "cpu_delta":
                        round(
                            max(0, cpu_delta),
                            6
                        ),

                    "read_bytes":
                        read_bytes,

                    "write_bytes":
                        write_bytes,

                    "read_bytes_per_sec":
                        round(
                            max(0, read_rate),
                            2
                        ),

                    "write_bytes_per_sec":
                        round(
                            max(0, write_rate),
                            2
                        )
                })

            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                psutil.ZombieProcess
            ):
                pass

        time.sleep(1)

except KeyboardInterrupt:

    print("\nStopping collection...")

    df = pd.DataFrame(data)

    df.to_csv(
        "disk_anomaly.csv",
        index=False
    )

    print(
        f"Saved {len(df)} observations."
    )

    print(
        "disk_anomaly.csv"
    )
