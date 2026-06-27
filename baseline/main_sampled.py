from sampled_runs import run_raw_sampling, run_L1_sampling, run_sig49_sampling, run_sig157_sampling

# python3 /project_antwerp/baseline/averaging_baseline/main_sampled.py

if __name__ == "__main__":

    seeds = [11, 222, 333, 4444, 55555]

    with open("averaging_sampled_baselines.txt", "w") as f:
        
        #######################################################################################
        # sampled runs
        #######################################################################################

        f.write("#"* 20 + "\n")
        f.write("Sampled Runs:\n")
        f.write("-"* 10 + "\n")
        f.flush()

        # raw
        try:
            f.write("Raw:\n")
            raw_sampled_aurocs = []
            for seed in seeds:
                auroc = run_raw_sampling(seed)
                raw_sampled_aurocs.append(auroc)
                f.write(f"Raw - Seed {seed}: {auroc:.4f}\n")
            f.write(f"Raw - Average: {sum(raw_sampled_aurocs) / len(raw_sampled_aurocs):.4f}\n")
            f.write("-"* 10 + "\n")
            f.flush()
        except Exception as e:
            print(f"Raw sampled runs failed with error: {e}\n")
            f.write(f"Raw sampled runs failed with error: {e}\n")
            f.write("-"* 10 + "\n")

        # L1
        try:
            f.write("L1:\n")
            L1_sampled_aurocs = []
            for seed in seeds:
                auroc = run_L1_sampling(seed)
                L1_sampled_aurocs.append(auroc)
                f.write(f"L1 - Seed {seed}: {auroc:.4f}\n")
            f.write(f"L1 - Average: {sum(L1_sampled_aurocs) / len(L1_sampled_aurocs):.4f}\n")
            f.write("-"* 10 + "\n")
            f.flush()
        except Exception as e:
            print(f"L1 sampled runs failed with error: {e}\n")
            f.write(f"L1 sampled runs failed with error: {e}\n")
            f.write("-"* 10 + "\n")

        # sig49
        try:
            f.write("sig49:\n")
            sig49_sampled_aurocs = []
            for seed in seeds:
                auroc = run_sig49_sampling(seed)
                sig49_sampled_aurocs.append(auroc)
                f.write(f"sig49 - Seed {seed}: {auroc:.4f}\n")
            f.write(f"sig49 - Average: {sum(sig49_sampled_aurocs) / len(sig49_sampled_aurocs):.4f}\n")
            f.write("-"* 10 + "\n")
            f.flush()
        except Exception as e:
            print(f"sig49 sampled runs failed with error: {e}\n")
            f.write(f"sig49 sampled runs failed with error: {e}\n")
            f.write("-"* 10 + "\n")

        # sig157
        try:
            sig157_sampled_aurocs = []
            for seed in seeds:
                auroc = run_sig157_sampling(seed)
                sig157_sampled_aurocs.append(auroc)
                f.write(f"sig157 - Seed {seed}: {auroc:.4f}\n")
            f.write(f"sig157 - Average: {sum(sig157_sampled_aurocs) / len(sig157_sampled_aurocs):.4f}\n")
            f.write("-"* 10 + "\n")
            f.flush()
        except Exception as e:
            print(f"sig157 sampled runs failed with error: {e}\n")
            f.write(f"sig157 sampled runs failed with error: {e}\n")
            f.write("-"* 10 + "\n")
