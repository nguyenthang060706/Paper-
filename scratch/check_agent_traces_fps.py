report = open('fp_fn_report_smoke_200.txt', encoding='utf-8').read()
fps = [line for line in report.split('\n\n') if 'src=agent-traces' in line]
print(f"Total agent-traces FP: {len(fps)}")
for i, fp in enumerate(fps):
    safe_fp = fp[:160].encode('ascii', errors='backslashreplace').decode('ascii')
    print(f"[{i+1}] {safe_fp}")
