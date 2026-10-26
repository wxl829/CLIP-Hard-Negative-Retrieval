from datasets import load_dataset


dataset = load_dataset(
    "AnyModal/flickr30k"
)


print(dataset)


for split in dataset.keys():
    print(
        split,
        len(dataset[split])
    )


sample = dataset["train"][0]

print("\n字段:")
print(sample.keys())

print("\n第一条:")
for k,v in sample.items():
    if k != "image":
        print(k, ":", v)