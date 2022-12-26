from argparse import ArgumentParser
import tqdm


import faiss
import torch
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler



from src.backbone.resnet import feature_extractor
from src.utils.dataloader import FlowersDataLoader


from src.indexer.faiss_indexer import get_faiss_indexer


def main():

    parser = ArgumentParser()
    parser.add_argument("--image_root", required=True, type=str)
    parser.add_argument("--device", required=False, type=str, default='cuda:0')
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    device = torch.device(args.device)
    batch_size = args.batch_size

    model = feature_extractor()
    model = model.to(device)

    dataset = FlowersDataLoader(args.image_root)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,batch_size=batch_size,sampler=sampler)

    flower_indexer = get_faiss_indexer()
    for indices, (images, image_paths) in enumerate(dataloader):
        images = images.to(device)
        features = model(images)
        # print(features.shape)
        flower_indexer.add(features.cpu().detach().numpy())
    faiss.write_index(flower_indexer, 'flowers.index.bin')

if __name__ == '__main__':
    main()
