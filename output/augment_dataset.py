import json
import base64
import random
import uuid
import pandas as pd
import os

print("Starting dataset augmentation...")

# 1. Generate Benign Base64 samples for Layer 3 Obfuscation
base64_benign_templates = [
    "echo '{b64}' | base64 -d > config.json",
    "kubectl create secret generic my-secret --from-literal=key='{b64}'",
    "data:image/png;base64,{b64}",
    "Authorization: Basic {b64}",
    "cat secret.txt | base64",
    "base64_decode('{b64}')",
    "let img = 'data:image/jpeg;base64,{b64}';",
    "const token = Buffer.from('{b64}', 'base64').toString('ascii');",
    "file_content_b64: '{b64}'",
    "export KUBECONFIG_B64='{b64}'",
    "Content-Transfer-Encoding: base64\n\n{b64}",
    "import base64; base64.b64decode('{b64}')"
]

benign_strings = [
    "{\"db_host\": \"localhost\", \"db_port\": 5432}",
    "user:password123",
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=", # 1x1 pixel png
    "https://api.example.com/v1/",
    "hello world",
    "ssh-rsa AAAAB3NzaC1yc2E... user@host",
    "{\"name\": \"test\", \"version\": \"1.0.0\"}",
    "import numpy as np; print(np.zeros(5))",
    "path/to/some/benign/file.txt",
    "test_user:test_password"
]

layer3_new_samples = []
for i in range(150):
    raw_str = random.choice(benign_strings)
    if raw_str.startswith("iVBORw"):
        b64_val = raw_str
    else:
        b64_val = base64.b64encode(raw_str.encode()).decode()
    
    action = random.choice(base64_benign_templates).replace("{b64}", b64_val)
    
    sample = {
        "action": action,
        "label": "benign",
        "session_id": f"benign-b64-{uuid.uuid4().hex[:8]}",
        "attack_type": "none", # Using string 'none' for consistency with other benign samples if applicable, or None. Let's use 'none' 
        "step_index": 0,
        "source_dataset": "synthetic-benign",
        "augmented": True
    }
    layer3_new_samples.append(sample)

with open("d:/Formal/Nigga/output/evo_pca_layer3_obfuscation.jsonl", "a", encoding="utf-8") as f:
    for s in layer3_new_samples:
        f.write(json.dumps(s) + "\n")

print(f"Added {len(layer3_new_samples)} benign base64 samples to Layer 3 Obfuscation.")


# 2. Generate Benign Bash/Tool Calls for Layer 2
layer2_templates = [
    "ls -la {dir}",
    "cat {file}",
    "grep '{term}' {file}",
    "python {script} {args}",
    "npm run {npm_cmd}",
    "git commit -m '{git_msg}'",
    "git push origin main",
    "docker build -t {img} .",
    "curl -s https://api.github.com/users/{user}",
    "tail -n 100 {file}",
    "chmod +x {file}",
    "mkdir -p {dir}",
    "rm -rf {dir}/tmp",
    "echo '{term}' > {file}",
    "vim {file}",
    "top -b -n 1",
    "ps aux | grep {term}",
    "df -h",
    "free -m",
    "tar -czvf archive.tar.gz {dir}",
    "cd {dir} && npm install",
    "kubectl get pods -n kube-system",
    "terraform apply -auto-approve",
    "sed -i 's/foo/bar/g' {file}",
    "awk '{{print $1}}' {file}"
]

dirs = ["/var/log", "/etc", "/home/user", "./src", "./build", "/tmp", "/opt/app", "/usr/local/bin", "~/project"]
files = ["config.json", "app.py", "package.json", "index.js", ".env.example", "README.md", "server.log", "data.csv"]
terms = ["error", "warning", "TODO", "import", "def", "function", "module", "class", "return"]
scripts = ["main.py", "train.py", "test.py", "setup.py", "manage.py", "script.py"]
args = ["--verbose", "--batch-size 32", "--epochs 10", "", "--help", "-v", "--output results.json"]
npm_cmds = ["build", "test", "start", "dev", "lint", "format"]
git_msgs = ["fix bug", "update readme", "add new feature", "refactor code", "initial commit", "WIP", "resolve merge conflict"]
imgs = ["myapp:latest", "ubuntu:20.04", "node:14-alpine", "nginx:stable", "postgres:13"]
users = ["octocat", "torvalds", "microsoft", "google", "apple"]

layer2_new_samples = []
for i in range(5000): # Generating 5000 samples to balance the 71% malicious 
    t = random.choice(layer2_templates)
    action = t.format(
        dir=random.choice(dirs),
        file=random.choice(files),
        term=random.choice(terms),
        script=random.choice(scripts),
        args=random.choice(args),
        npm_cmd=random.choice(npm_cmds),
        git_msg=random.choice(git_msgs),
        img=random.choice(imgs),
        user=random.choice(users)
    )
    
    sample = {
        "action": action,
        "label": "benign",
        "session_id": f"benign-bash-{uuid.uuid4().hex[:8]}",
        "attack_type": "none",
        "step_index": 0,
        "source_dataset": "synthetic-nl2bash-sim",
        "augmented": True
    }
    layer2_new_samples.append(sample)

with open("d:/Formal/Nigga/output/evo_pca_layer2.jsonl", "a", encoding="utf-8") as f:
    for s in layer2_new_samples:
        f.write(json.dumps(s) + "\n")

print(f"Added {len(layer2_new_samples)} benign bash samples to Layer 2.")

# 3. Update Parquet files
print("Updating Parquet files...")
for layer in ["layer2", "layer3_obfuscation"]:
    jsonl_path = f"d:/Formal/Nigga/output/evo_pca_{layer}.jsonl"
    parquet_path = f"d:/Formal/Nigga/output/evo_pca_{layer}.parquet"
    if os.path.exists(jsonl_path):
        df = pd.read_json(jsonl_path, lines=True)
        df.to_parquet(parquet_path, index=False)
        print(f"Updated {parquet_path} (Total rows: {len(df)})")
    else:
        print(f"Warning: {jsonl_path} not found.")

print("Done.")
