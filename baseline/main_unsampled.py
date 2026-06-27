from unsampled_runs import run_raw, run_L1, run_sig49, run_sig157

# python3 /project_antwerp/baseline/averaging_baseline/main_unsampled.py

if __name__ == "__main__":

    seeds = [11, 222, 333, 4444, 55555]

    with open("averaging_unsampled_baselines.txt", "w") as f:
        
        ###################################################################################
        # unsampled runs
        ###################################################################################

        f.write("#"* 20 + "\n")
        f.write("Unsampled Runs:\n")
        f.write("-"* 10 + "\n")
        f.flush()
        
        # raw
        try:
            f.write("Raw:\n")
            raw_aurocs = []
            for seed in seeds:
                auroc = run_raw(seed)
                raw_aurocs.append(auroc)
                f.write(f"Raw - Seed {seed}: {auroc:.4f}\n")
            f.write(f"Raw - Average: {sum(raw_aurocs) / len(raw_aurocs):.4f}\n")
            f.write("-"* 10 + "\n")
            f.flush()
        except Exception as e:
            print(f"Raw unsampled runs failed with error: {e}\n")
            f.write(f"Raw unsampled runs failed with error: {e}\n")
            f.write("-"* 10 + "\n")

        # L1
        try:
            f.write("L1:\n")
            L1_aurocs = []
            for seed in seeds:
                auroc = run_L1(seed)
                L1_aurocs.append(auroc)
                f.write(f"L1 - Seed {seed}: {auroc:.4f}\n")
            f.write(f"L1 - Average: {sum(L1_aurocs) / len(L1_aurocs):.4f}\n")
            f.write("-"* 10 + "\n")
            f.flush()
        except Exception as e:
            print(f"L1 unsampled runs failed with error: {e}\n")
            f.write(f"L1 unsampled runs failed with error: {e}\n")
            f.write("-"* 10 + "\n")

        # sig49
        try:
            f.write("sig49:\n")
            sig49_aurocs = []
            for seed in seeds:
                auroc = run_sig49(seed)
                sig49_aurocs.append(auroc)
                f.write(f"sig49 - Seed {seed}: {auroc:.4f}\n")
            f.write(f"sig49 - Average: {sum(sig49_aurocs) / len(sig49_aurocs):.4f}\n")
            f.write("-"* 10 + "\n")
            f.flush()
        except Exception as e:
            print(f"sig49 unsampled runs failed with error: {e}\n")
            f.write(f"sig49 unsampled runs failed with error: {e}\n")
            f.write("-"* 10 + "\n")

        # sig157
        try:
            f.write("sig157:\n")
            sig157_aurocs = []
            for seed in seeds:
                auroc = run_sig157(seed)
                sig157_aurocs.append(auroc)
                f.write(f"sig157 - Seed {seed}: {auroc:.4f}\n")
            f.write(f"sig157 - Average: {sum(sig157_aurocs) / len(sig157_aurocs):.4f}\n")
            f.write("-"* 10 + "\n")
            f.flush()
        except Exception as e:
            print(f"sig157 unsampled runs failed with error: {e}\n")
            f.write(f"sig157 unsampled runs failed with error: {e}\n")
            f.write("-"* 10 + "\n")

        
