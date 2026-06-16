from app.dataset_loader import satellitePose
from torch.utils.data import DataLoader

# json label path
json_path = r"D:\Data centr\IMG_data\satellite_pose\speed\train.json"

# image folder path
image_path = r"D:\Data centr\IMG_data\satellite_pose\speed\images\train"


if __name__ == "__main__":

    # create dataset object
    dataset = satellitePose(json_path, image_path)

    print("Total samples:", len(dataset))

    image, pose = dataset[0]

    print("Image shape:", image.shape)
    print("Pose vector:", pose)


    # create dataloader (batch generator)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    images, poses = next(iter(loader))

    print("Batch image shape:", images.shape)
    print("Batch pose shape:", poses.shape)