from huggingface_hub import HfApi, login

login(token="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")

api = HfApi()

PDE = "active_matter"
base_path = f"/glade/work/gaccarino/data/datasets/{PDE}/data"

try:
    url = api.upload_folder(
        folder_path=base_path,
        repo_id="gabrieleaccarino/waveletflux-dataset",
        repo_type="dataset",
        path_in_repo=f"{PDE}",
        commit_message=f"Upload {PDE} dataset (train/valid/test)"
    )
    print(f"Done: {url}")
except Exception as e:
    print(f"Failed: {e}")