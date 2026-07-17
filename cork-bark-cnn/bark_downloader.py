import kagglehub
# this never worked btw

path = kagglehub.dataset_download("devashishbhake01/barknet-large/CHR", output_dir="./cork_oak_dataset/Training/0_Healthy")

print("Path to dataset files:", path)