from typing import List, Dict

import os
import sys
import zipfile
import requests  # type: ignore


VAULT_INFO = {
    "og_marl": {
        "smac_v1": {
            "3m": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v1/3m.zip"
            },
            "8m": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v1/8m.zip"
            },
            "5m_vs_6m": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v1/5m_vs_6m.zip"
            },
            "2s3z": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v1/2s3z.zip"
            },
            "3s5z_vs_3s6z": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v1/3s5z_vs_3s6z.zip"
            },
        },
        "smac_v2": {
            "terran_5_vs_5": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v2/terran_5_vs_5.zip"
            },
            "terran_10_vs_10": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v2/terran_10_vs_10.zip"
            },
            "zerg_5_vs_5": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v2/zerg_5_vs_5.zip"
            },
        },
        "mamujoco": {
            "2halfcheetah": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/mamujoco/2halfcheetah.zip"
            },
            "2ant": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/mamujoco/2ant.zip"
            },
        },
        "gymnasium_mamujoco": {
            "2ant": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/gymnasium_mamujoco/2ant.zip"
            },
            "2halfcheetah": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/gymnasium_mamujoco/2halfcheetah.zip"
            },
            "2walker": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/gymnasium_mamujoco/2walker.zip"
            },
            "3hopper": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/gymnasium_mamujoco/3hopper.zip"
            },
            "4ant": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/gymnasium_mamujoco/4ant.zip"
            },
            "6halfcheetah": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/gymnasium_mamujoco/6halfcheetah.zip"
            },
        },
        "flatland": {
            "20_trains": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/flatland/20_trains.zip"
            },
            "30_trains": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/flatland/30_trains.zip"
            },
        },
    },
    "cfcql": {
        "smac_v1": {
            "6h_vs_8z": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/cfcql/smac_v1/6h_vs_8z.zip"
            },
            "3s_vs_5z": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/cfcql/smac_v1/3s_vs_5z.zip"
            },
            "5m_vs_6m": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/cfcql/smac_v1/5m_vs_6m.zip"
            },
            "2s3z": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/cfcql/smac_v1/2s3z.zip"
            },
        },
    },
    "alberdice": {
        "rware": {
            "small-2ag": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/alberdice/rware/small-2ag.zip"
            },
            "small-4ag": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/alberdice/rware/small-4ag.zip"
            },
            "small-6ag": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/alberdice/rware/small-6ag.zip"
            },
            "tiny-2ag": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/alberdice/rware/tiny-2ag.zip"
            },
            "tiny-4ag": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/alberdice/rware/tiny-4ag.zip"
            },
            "tiny-6ag": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/alberdice/rware/tiny-6ag.zip"
            },
        },
    },
    "omar": {
        "mpe": {
            "simple_spread": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/omar/mpe/simple_spread.zip"
            },
            "simple_tag": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/omar/mpe/simple_tag.zip"
            },
            "simple_world": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/omar/mpe/simple_world.zip"
            },
        },
        "mamujoco": {
            "2halfcheetah": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/omar/mamujoco/2halfcheetah.zip"
            },
        },
    },
    "omiga": {
        "smac_v1": {
            "2c_vs_64zg": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/omiga/smac_v1/2c_vs_64zg.zip"
            },
            "6h_vs_8z": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/omiga/smac_v1/6h_vs_8z.zip"
            },
            "5m_vs_6m": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/omiga/smac_v1/5m_vs_6m.zip"
            },
            "corridor": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/omiga/smac_v1/corridor.zip"
            },
        },
        "mamujoco": {
            "6halfcheetah": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/omiga/mamujoco/6halfcheetah.zip"
            },
            "2ant": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/omiga/mamujoco/2ant.zip"
            },
            "3hopper": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/omiga/mamujoco/3hopper.zip"
            },
        },
    },
}


def print_download_options() -> Dict[str, Dict]:
    print("VAULT_INFO:")
    for source in VAULT_INFO.keys():
        print(f"\t {source}")
        for env in VAULT_INFO[source].keys():
            print(f"\t \t {env}")
            for scenario in VAULT_INFO[source][env].keys():
                print(f"\t \t \t {scenario}")
    return VAULT_INFO


def download_and_unzip_vault(
    dataset_source: str,
    env_name: str,
    scenario_name: str,
    dataset_base_dir: str = "/data",
    dataset_download_url: str = "",
) -> str:
    if check_directory_exists_and_not_empty(
        f"{dataset_base_dir}/{dataset_source}/{env_name}/{scenario_name}.vlt"
    ):
        print(
            f"Vault '{dataset_base_dir}/{dataset_source}/{env_name}/{scenario_name}' already exists."  # noqa
        )
        return f"{dataset_base_dir}/{dataset_source}/{env_name}/{scenario_name}.vlt"

    if len(dataset_download_url) == 0:
        dataset_download_url = VAULT_INFO[dataset_source][env_name][scenario_name]["url"]

    try:
        response = requests.get(dataset_download_url, stream=True)
    except Exception:
        print(
            "Dataset from "
            + str(dataset_download_url)
            + " could not be downloaded. Try entering a different URL, or removing the part which auto-downloads."  # noqa
        )
        return f"{dataset_base_dir}/{dataset_source}/{env_name}/{scenario_name}.vlt"

    total_length = response.headers.get("content-length")

    os.makedirs(f"{dataset_base_dir}/tmp/", exist_ok=True)
    os.makedirs(f"{dataset_base_dir}/{dataset_source}/{env_name}/", exist_ok=True)

    zip_file_path = f"{dataset_base_dir}/tmp/tmp_dataset.zip"
    extraction_path = f"{dataset_base_dir}/{dataset_source}/{env_name}"

    with open(zip_file_path, "wb") as file:
        if total_length is None:
            file.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)  # type: ignore
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                file.write(data)
                done = int(50 * dl / total_length)  # type: ignore
                sys.stdout.write("\r[%s%s]" % ("=" * done, " " * (50 - done)))
                sys.stdout.flush()

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extraction_path)

    os.remove(zip_file_path)

    return f"{dataset_base_dir}/{dataset_source}/{env_name}/{scenario_name}.vlt"


def check_directory_exists_and_not_empty(path: str) -> bool:
    if os.path.exists(path) and os.path.isdir(path):
        if not os.listdir(path):
            return False
        else:
            return True
    else:
        return False


def get_available_uids(rel_vault_path: str) -> List[str]:
    vault_uids = sorted(
        next(os.walk(rel_vault_path))[1],
        reverse=True,
    )
    return vault_uids
