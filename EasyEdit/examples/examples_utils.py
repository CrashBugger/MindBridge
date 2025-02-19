def get_best_gpu(n=1):
    import subprocess

    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                            stdout=subprocess.PIPE)
    gpu_info = result.stdout.decode('utf-8').strip().split('\n')

    gpus = sorted([(int(index), int(memory_free)) for index, memory_free in (gpu.split(', ') for gpu in gpu_info)],
                  key=lambda x: x[1],
                  reverse=True)

    best_gpus = [gpu[0] for gpu in gpus[:n]]
    print(best_gpus)
    return ",".join(str(gpu) for gpu in best_gpus)


def clean_dataset(dataset):
    new_dataset = []
    for i in dataset:
        if i["alt"] != "":
            new_dataset.append(i)
    print(f"{len(new_dataset)} <- {len(dataset)}")
    return new_dataset
