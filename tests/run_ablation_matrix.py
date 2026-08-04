import os
import sys

# Setup environment to load the pipeline properly
os.environ["EVO_PCA_ENABLE_LSTM_TIER05"] = "true"

# Define matrix configs
CONFIGS = [
    {"name": "lstm_off_flagfix_on", "env": {"EVO_PCA_ENABLE_LSTM_TIER05": "false"}},
    {"name": "lstm_on_flagfix_on", "env": {"EVO_PCA_ENABLE_LSTM_TIER05": "true"}},
]

import subprocess

def run_matrix():
    print("Starting Benchmark Matrix Execution (2x2 Ablation Study)...")
    
    # We already fixed skip_rce in tier05.py, so "flagfix_on" is persistent in code.
    for cfg in CONFIGS:
        print(f"\n[{cfg['name']}] Preparing run...")
        
        # Setup env
        env = os.environ.copy()
        for k, v in cfg['env'].items():
            env[k] = v
            print(f"  Set {k} = {v}")
            
        csv_out = f"summary_{cfg['name']}.csv"
        
        # We need to run run_benchmark.py
        # run_benchmark.py writes to agentdojo_benchmark_summary.csv by default,
        # so we'll run it, then rename the file.
        
        cmd = [sys.executable, "tests/run_benchmark.py"]
        print(f"  Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, env=env, cwd="d:/DEMO_GROUP_1", capture_output=True, text=True, check=True)
            print(f"  Run successful!")
            
            # Rename output
            if os.path.exists("d:/DEMO_GROUP_1/agentdojo_benchmark_summary.csv"):
                os.rename(
                    "d:/DEMO_GROUP_1/agentdojo_benchmark_summary.csv", 
                    f"d:/DEMO_GROUP_1/{csv_out}"
                )
                print(f"  Saved results to {csv_out}")
            else:
                print(f"  WARNING: Expected output agentdojo_benchmark_summary.csv not found.")
                
        except subprocess.CalledProcessError as e:
            print(f"  ERROR running benchmark for {cfg['name']}: {e}")
            print(e.stderr)

if __name__ == "__main__":
    run_matrix()
