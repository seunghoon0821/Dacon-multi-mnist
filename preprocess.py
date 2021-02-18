from torchvision import transforms
import albumentations as A
import albumentations.pytorch


# transforms_train = transforms.Compose([
#     # transforms.RandomHorizontalFlip(p=0.5),
#     # transforms.RandomVerticalFlip(p=0.5),
#     transforms.RandomRotation(degrees=90),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         [0.485, 0.456, 0.406],
#         [0.229, 0.224, 0.225]
#     )
# ])

# transforms_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(
#         [0.485, 0.456, 0.406],
#         [0.229, 0.224, 0.225]
#     )
# ])


transforms_train = A.Compose([
    # A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    # A.ElasticTransform(p=0.5),
    # A.GridDistortion(p=0.5),
    A.Normalize(),
    A.pytorch.transforms.ToTensor()
])

transforms_test = A.Compose([
    A.Normalize(),
    A.pytorch.transforms.ToTensor()
])