#  import ray
import polars as pl
import numpy as np
import os
import itertools

data_path = "data"
#  data_path = "data.2024.01.26.17.25"
#  data_path = "data.2024.01.26.17.36"
#  data_path = "data.2024.01.26.18.46"
#  data_path = "data.2024.01.27.05.50"
#  data_path = "data.2024.01.27.10.20"
#  data_path = "data.2024.01.27.11.22"
#  data_path = "data.2024.01.27.14.36"
#  data_path = "data.2024.01.27.15.17"
#  data_path = "data.2024.01.27.17.48"
data_path = "data.2024.01.28.00.00"


def bootstrap_std_of_mean(nums, n_bootstrap_samples=1000):
    # Array to store the mean of each bootstrap sample
    bootstrap_means = np.zeros(n_bootstrap_samples)

    # Generate bootstrap samples and compute their means
    for i in range(n_bootstrap_samples):
        bootstrap_sample = np.random.choice(nums, size=len(nums), replace=True)
        bootstrap_means[i] = np.mean(bootstrap_sample)

    # The standard deviation of the bootstrap means is an estimate of the std of the original mean
    std_of_mean = np.std(bootstrap_means)
    return std_of_mean


def mu_sigma_std(nums):
    nums = np.array(nums)
    n = len(nums)
    mu = nums.mean()
    sigma = nums.std()
    std = sigma / np.sqrt(n)
    #  std = bootstrap_std_of_mean(nums, n_bootstrap_samples=100)
    return mu, sigma, std


def mu_sigma_std_from_sums(sums, nums):
    #  $X\sim N(\mu,\sigma)$
    #  $S_i=\sum_{j=1}^{n_i} X_{i,j}$
    #  best mean:
    #  $\hat{\mu}=\frac{\sum_{i=1}^m S_m}{\sum_{i=1}^m n_i}
    #      variance:
    #          $Var(\hat{\mu})=\frac{\sigma^2}{\sum_{i=1}^m n_i}$
    #  best variance:
    #      $\hat{\sigma^2}=\frac{1}{m-1}[\sum_{i=1}^m\frac{S_i^2}{n_i}-\frac{(\sum_{i=1}^m S_i)^2}{\sum_{i=1}^m n_i}]$
    #      $\hat{\sigma^2}=\frac{1}{m-1}[\sum_{i=1}^m n_i(\frac{S_i}{n_i})^2-\sum_{i=1}^m n_i(\frac{\sum_{i=1}^m S_i}{\sum_{i=1}^m n_i})^2]$
    #      $\hat{\sigma^2}=\frac{1}{m-1}[\sum_{i=1}^m n_i(\frac{S_i}{n_i})^2-\sum_{i=1}^m n_i\hat{\mu}^2]$
    assert len(sums) == len(nums)
    sums = np.array(sums)
    nums = np.array(nums)
    mask = nums > 0
    sums = sums[mask]
    nums = nums[mask]
    mu = sums.sum() / nums.sum()
    sigma2 = ((sums**2 / nums).sum() - nums.sum() * mu**2) / (len(sums) - 1)
    sigma = np.sqrt(sigma2)
    std = np.sqrt(sigma2 / nums.sum())
    return mu, sigma, std


def ration_mu_std(mu1, std1, mu2, std2):
    mu = mu1 / mu2
    std = mu * np.sqrt((std1 / mu1) ** 2 + (std2 / mu2) ** 2)
    return mu, std


def format_mu_std(mu, std, digit=None, latex=False):
    if digit is None:
        # infer_digit based on std
        # if std starts with 1, keep another digit
        # if std starts with 2 or more, keep to that digit
        first_digit = -int(np.log10(std)) + 1
        if std * 10**first_digit >= 2:
            effective_digit = first_digit
        else:
            effective_digit = first_digit + 1
        digit = max(0, effective_digit)
    if not latex:
        s = f"{mu:.{digit}f}±{std:.{digit}f}"
    else:
        s = f"${mu:.{digit}f} \pm {std:.{digit}f}$"
    return s


def analysis_translation_scan_n():
    test_strs = os.listdir(
        os.path.join(
            os.path.dirname(__file__), "data_root", data_path, "translation_scan_n"
        )
    )
    for test_str in test_strs:
        ds = pl.read_parquet(
            os.path.join(
                os.path.dirname(__file__),
                "data_root",
                data_path,
                "translation_scan_n",
                test_str,
                "*",
            )
        )
        keys = ["method", "n"]
        ptt_ds = (
            ds.select(
                [
                    *keys,
                    "gen_seq_lens",
                    "t_got_first_output",
                    "t_got_last_output",
                ]
            )
            .group_by(keys)
            .agg(
                nums=pl.col("gen_seq_lens").list.slice(1, None).list.sum(),
                sums=pl.col("t_got_last_output") - pl.col("t_got_first_output"),
            )
            .with_columns(
                tobeunnest=pl.struct(["nums", "sums"]).map_elements(
                    lambda x: {
                        k: v
                        for k, v in zip(
                            ["ptt_mu", "ptt_sigma", "ptt_std"],
                            mu_sigma_std_from_sums(**x),
                        )
                    }
                )
            )
            .unnest("tobeunnest")
        ).drop(["nums", "sums"])
        atn_ds = (
            ds.select(
                [
                    *keys,
                    "gen_seq_lens",
                ]
            )
            .group_by(keys)
            .agg(
                nums=pl.col("gen_seq_lens").flatten(),
            )
            .with_columns(
                tobeunnest=pl.struct(["nums"]).map_elements(
                    lambda x: {
                        k: v
                        for k, v in zip(
                            ["atn_mu", "atn_sigma", "atn_std"],
                            mu_sigma_std(**x),
                        )
                    }
                )
            )
            .unnest("tobeunnest")
        ).drop(["nums"])
        out1_ds = ptt_ds.join(atn_ds, on=keys).sort(by=keys)
        with pl.Config(tbl_rows=out1_ds.height):
            print(test_str)
            print(out1_ds)

        # improvement of method=tmc compared to method=mc
        def compute_tmc_mc_ratio(tmcmc_out1_ds, mu_std_cols):
            mu_pivot_ds = tmcmc_out1_ds.pivot(
                values=mu_std_cols[0], index="n", columns="method"
            ).rename(lambda x: x + "_mu" if x != "n" else x)
            #  ).rename(lambda column_name: "c" + column_name[1:])
            std_pivot_ds = tmcmc_out1_ds.pivot(
                values=mu_std_cols[1], index="n", columns="method"
            ).rename(lambda x: x + "_std" if x != "n" else x)
            #  ).rename(lambda column_name: "d" + column_name[1:])
            pivot_ds = mu_pivot_ds.join(std_pivot_ds, on="n")
            return (
                pivot_ds.with_columns(
                    tobeunnest=pl.struct(
                        mu1=pivot_ds["tmc_mu"],
                        std1=pivot_ds["tmc_std"],
                        mu2=pivot_ds["mc_mu"],
                        std2=pivot_ds["mc_std"],
                    ).map_elements(
                        lambda x: {
                            k: v
                            for k, v in zip(
                                ["ratio_" + x for x in mu_std_cols],
                                ration_mu_std(**x),
                            )
                        }
                    )
                )
                .select(["n", "tobeunnest"])
                .unnest("tobeunnest")
            )

        tmcmc_out1_ds = out1_ds.filter(pl.col("method").is_in(["tmc", "mc"]))
        ratio_atn = compute_tmc_mc_ratio(tmcmc_out1_ds, ["atn_mu", "atn_std"])
        ratio_ptt = compute_tmc_mc_ratio(tmcmc_out1_ds, ["ptt_mu", "ptt_std"])
        out2_ds = (
            ratio_atn.join(ratio_ptt, on="n")
            .with_columns(
                ptt_decrease_ratio_mu=1 - pl.col("ratio_ptt_mu"),
                ptt_decrease_ratio_std=pl.col("ratio_ptt_std"),
                atn_increase_ratio_mu=pl.col("ratio_atn_mu") - 1,
                atn_increase_ratio_std=pl.col("ratio_atn_std"),
            )
            .drop(["ratio_ptt_mu", "ratio_ptt_std", "ratio_atn_mu", "ratio_atn_std"])
        )
        print(out2_ds)
        #  atn_sum_ds = (
        #      out2_ds.select(["n", "atn_increase_ratio_mu", "atn_increase_ratio_std"])
        #      .with_columns(
        #          tobeunnest=pl.struct(
        #              mu=pl.col("atn_increase_ratio_mu"),
        #              std=pl.col("atn_increase_ratio_std"),
        #          ).map_elements(lambda x: {"improvement": format_mu_std(**x)})
        #      )
        #      .unnest("tobeunnest")
        #      .drop(["atn_increase_ratio_mu", "atn_increase_ratio_std"])
        #  )
        atn_sum_ds = (
            out1_ds.filter(pl.col("method").is_in(["tmc", "mc"]))
            .select(["n", "method", "atn_mu", "atn_std"])
            .with_columns(
                atn=pl.struct(
                    mu=pl.col("atn_mu"),
                    std=pl.col("atn_std"),
                ).map_elements(lambda x: format_mu_std(**x, digit=3))
            )
            .drop(["atn_mu", "atn_std"])
            .pivot(values="atn", index="n", columns="method")
            .rename({"tmc": "ASpS", "mc": "SpS"})
            .join(
                out2_ds.select(["n", "atn_increase_ratio_mu", "atn_increase_ratio_std"])
                .with_columns(
                    improvement=pl.struct(
                        mu=pl.col("atn_increase_ratio_mu") * 100,
                        std=pl.col("atn_increase_ratio_std") * 100,
                    ).map_elements(lambda x: format_mu_std(**x, digit=1))
                )
                .drop(["atn_increase_ratio_mu", "atn_increase_ratio_std"]),
                on="n",
            )
        )
        print(atn_sum_ds)


if __name__ == "__main__":
    analysis_translation_scan_n()
