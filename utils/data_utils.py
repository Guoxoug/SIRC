import torch
import torchvision as tv
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torchvision.datasets.folder import ImageFolder
import os
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

TRAIN_DATASETS = [
    "imagenet",
    "imagenet200"
]

# Transforms for when imaged data is loaded -----------------------------------

# taken from https://arxiv.org/pdf/1512.03385.pdf
# also used in https://arxiv.org/abs/1608.06993
# contains transforms for each dataset, norm is factored out
# so that when dataset is OOD, the training values can be used
# https://github.com/pytorch/vision/issues/39#issuecomment-403701432
# https://paperswithcode.github.io/torchbench/imagenet/
# not necessarily the same as all papers

IMAGENET_TRANSFORMS = {
    "train": tv.transforms.Compose(
        [
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
        ]
    ),
    "test": tv.transforms.Compose(
        [
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
           
        ]
    ),
    "norm":  tv.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
}

# PIL image (0-255) mapped to [0,1], then mean subtracted and std divided 
imagenet_mean=torch.tensor([0.485, 0.456, 0.406])
imagenet_std=torch.tensor([0.229, 0.224, 0.225])



# images are 150x150
# scale using a better method
COLORECTAL_TRANSFORMS ={
    "train": None,
    "test": tv.transforms.Compose(
        [
            tv.transforms.Resize(224, interpolation=InterpolationMode.LANCZOS),
            tv.transforms.ToTensor(),

        ]
    ),
    "norm":  tv.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
}



# adapting datasets from 
# https://github.com/daintlab/unknown-detection-benchmarks/blob/main/dataloader.py



DATASET_NAME_TRANSFORM_MAPPING = {
    "imagenet": IMAGENET_TRANSFORMS,
    "imagenet200": IMAGENET_TRANSFORMS,
    "imagenet-o": IMAGENET_TRANSFORMS,
    "caltech256": IMAGENET_TRANSFORMS,
    "near-imagenet200": IMAGENET_TRANSFORMS,
    "inaturalist": IMAGENET_TRANSFORMS,
    "imagenet-noise": IMAGENET_TRANSFORMS,
    "colorectal": COLORECTAL_TRANSFORMS,
    "colonoscopy": IMAGENET_TRANSFORMS,
    "openimage-o": IMAGENET_TRANSFORMS,
    "textures": IMAGENET_TRANSFORMS,
    "spacenet": IMAGENET_TRANSFORMS
}


# for tables/plots
DATA_NAME_MAPPING = {
    "imagenet200": "ImageNet-200",
    "near-imagenet200": "Near-ImageNet-200",
    "caltech256": "Caltech-45",
    "inaturalist": "iNaturalist",
    "imagenet-noise": "Noise",
    "colorectal": "Colorectal",

    # this is a bit of a hack for when generating tables from .csv files
    "errFPR@95": "ID\\xmark",
    "errROC": "ID\\xmark",

    "openimage-o": "Openimage-O",
    "textures":"Textures",
    "colonoscopy":"Colonoscopy",
    "imagenet-o": "ImageNet-O",
    "spacenet": "SpaceNet",
}


# get transforms for pre-processing
def get_preprocessing_transforms(
    dataset_name, id_dataset_name=None
) -> dict:
    """Get preprocessing transforms for a dataset.
    
    If the dataset is OOD then the ID/training set's normalisation values will
    be used (as if preprocessing is part of network input layer).
    """
    dataset_transforms = DATASET_NAME_TRANSFORM_MAPPING[dataset_name]
    
    # ID data, use its own normalisation
    if id_dataset_name is None:
        transforms = {
            "train": tv.transforms.Compose(
                [
                    dataset_transforms["train"],
                    dataset_transforms["norm"]
                ]
            ),
            "test": tv.transforms.Compose(
                [
                    dataset_transforms["test"],
                    dataset_transforms["norm"]
                ]
            )
        }
    else:

        # use the in distribution/training set's values for testing ood
        id_dataset_transforms = DATASET_NAME_TRANSFORM_MAPPING[id_dataset_name]
        transforms = {
            "train": tv.transforms.Compose(
                [
                    dataset_transforms["train"],
                    id_dataset_transforms["norm"]
                ]
            ),
            "test": tv.transforms.Compose(
                [
                    dataset_transforms["test"],
                    id_dataset_transforms["norm"]
                ]
            )
        }
    return transforms

# Data object -----------------------------------------------------------------
class Data:
    """Class that contains a datasets + loaders as well as information
    about the dataset, e.g. #samples, transforms for data augmentation.
    Allows the division of the training set into train and validation sets.
    """
    def __init__(
        self,
        name: str, 
        datapath: str,
        download=False,
        batch_size=64,
        test_batch_size=None,
        num_workers=8,
        drop_last=False,
        transforms={"train":None, "test":None},
        target_transforms={"train": None, "test": None},
        val_size=0,
        num_classes=None,
        test_only=False, 
        test_shuffle=False,
        **data_kwargs
    ) -> None:
        self.name = name
        self.datapath = datapath
        self.download = download
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size if (
            test_batch_size is not None
        ) else batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last 
        self.transforms = transforms

        # stays around in namespace between objects for some reason
        self.target_transforms = target_transforms.copy() 
        self.val_size = val_size
        self.num_classes = num_classes # this overwrites defaults
        self.test_only = test_only
        self.test_shuffle = test_shuffle
        self.data_kwargs = data_kwargs


        # get datasets and dataloaders

        # training/in distribution sets ---------------------------------------
        
        if self.name == "imagenet":

            self.num_classes = 1000 if (
                self.num_classes is None
            ) else self.num_classes

            # test set/loader
            # for imagenet use val as test
            # imagenet must be previously downloaded
            # torchvision uses tar files
            try:
                self.test_set = tv.datasets.ImageNet(
                    root=self.datapath,
                    split="val",
                    transform=self.transforms["test"],
                    target_transform=self.target_transforms["test"]
                )

            # normal folders
            except:
                try:
                    self.test_set = tv.datasets.ImageFolder(
                        root=os.path.join(self.datapath, "validation"),
                        transform=self.transforms["test"],
                        target_transform=self.target_transforms["test"]
                    )
                except:
                    self.test_set = tv.datasets.ImageFolder(
                        root=os.path.join(self.datapath, "val"),
                        transform=self.transforms["test"],
                        target_transform=self.target_transforms["test"]
                    )

            self.test_loader = DataLoader(
                self.test_set,
                batch_size=self.test_batch_size,
                shuffle=self.test_shuffle,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                pin_memory=True
            )

            # train
            if not self.test_only:
                try:
                    self.train_set = tv.datasets.ImageNet(
                        root=self.datapath,
                        split="train",
                        transform=self.transforms["train"],
                        target_transform=self.target_transforms["train"]
                    )
                except:
                    self.train_set = tv.datasets.ImageFolder(
                        root=os.path.join(self.datapath, "train"),
                        transform=self.transforms["train"],
                        target_transform=self.target_transforms["train"]
                    )

                # as we use the validation set as a test set
                # this allows us to segment out a validation set should
                # we need it
                # in this case the validation data set will point to 
                # all training images however, even if the loader only loads
                # specific indices
                indices = torch.randperm(len(self.train_set))
                idx_path = os.path.join(self.datapath, 'index.pth')
                if os.path.exists(idx_path):
                    print('!!!!!! Load train_set_index !!!!!!')
                    indices = torch.load(idx_path)
                else:
                    try:
                        print('!!!!!! Save train_set_index !!!!!!')
                        torch.save(indices, idx_path)
                    except:
                        print("couldn't save indices")


                if val_size > 0:
                    assert (
                        val_size <= len(self.train_set)
                    ), "val size larger than training set"

                    # train set with test transforms
                    try:
                        self.val_set = tv.datasets.ImageNet(
                            root=self.datapath,
                            split="train",
                            transform=self.transforms["test"],
                            target_transform=self.target_transforms["test"]
                        )
                    except:
                        self.test_set = tv.datasets.ImageFolder(
                            root=os.path.join(self.datapath, "train"),
                            transform=self.transforms["test"],
                            target_transform=self.target_transforms["test"]
                        )

                    # train/val split
                    self.train_indices = indices[0:-val_size]
                    self.val_indices = indices[val_size:]

                    self.val_loader = DataLoader(
                        self.val_set,
                        batch_size=self.test_batch_size,
                        num_workers=self.num_workers,
                        drop_last=self.drop_last,
                        sampler=SequentialSampler(
                            self.val_indices
                        ),
                        pin_memory=True
                    )

                else:
                    self.train_indices = indices

                self.train_loader = DataLoader(
                    self.train_set,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    drop_last=self.drop_last,
                    sampler=SubsetRandomSampler(
                        self.train_indices
                    ),
                    pin_memory=True
                )

        if self.name == "imagenet200":
            self.num_classes = 200 if (
                self.num_classes is None
            ) else self.num_classes

            # test set/loader
            self.test_set = ImageFolder(
                root=os.path.join(
                    self.datapath,
                    'test/imagenet200/labels/'
                ),
                transform=self.transforms["test"],
                target_transform=self.target_transforms["test"]
            )

            self.test_loader = DataLoader(
                self.test_set,
                batch_size=self.test_batch_size,
                shuffle=self.test_shuffle,
                num_workers=self.num_workers,
                drop_last=self.drop_last
            )

            # train
            if not self.test_only:
                self.train_set = ImageFolder(
                    root=os.path.join(
                        self.datapath,
                        'train/imagenet200/'
                    ),
                    transform=self.transforms["train"],
                    target_transform=self.target_transforms["train"]
                )

                # validation set is bugged, one of the folders is empty
                self.train_loader = DataLoader(
                    self.train_set,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    drop_last=self.drop_last,
                    shuffle=True
                )
        
        # OOD datasets --------------------------------------------------------
        if self.name == "imagenet-o":

            self.test_only = True
            print(f"{self.name} is only available for testing/OOD detection")

            self.test_set = tv.datasets.ImageFolder(
                root=self.datapath,
                transform=self.transforms["test"],
                target_transform=self.target_transforms["test"]
            )

            # this set is only 2000  images
            self.test_loader = DataLoader(
                self.test_set,
                batch_size=self.test_batch_size,
                shuffle=self.test_shuffle,
                num_workers=self.num_workers,
                drop_last=self.drop_last
            )
        # note, for datasets from 
        # https://github.com/daintlab/unknown-detection-benchmarks
        # datapath passed to Data class should be to "imagenet-bench"
        if "near-imagenet200" in self.name:

            self.test_only = True
            print(f"{self.name} is only available for testing/OOD detection")
            self.test_set = tv.datasets.ImageFolder(
                root=os.path.join(
                    self.datapath,
                    'test/near-imagenet200/labels/'
                ),
                transform=self.transforms["test"],
                target_transform=self.target_transforms["test"]
            )

            self.test_loader = DataLoader(
                self.test_set,
                batch_size=self.test_batch_size,
                shuffle=self.test_shuffle,
                num_workers=self.num_workers,
                drop_last=self.drop_last
            )

        if self.name == "caltech256":
            self.test_only = True
            print(f"{self.name} is only available for testing/OOD detection")
            self.test_set = tv.datasets.ImageFolder(
                root=os.path.join(
                    self.datapath,
                    'test/caltech256/labels/'
                ),
                transform=self.transforms["test"],
                target_transform=self.target_transforms["test"]
            )

            self.test_loader = DataLoader(
                self.test_set,
                batch_size=self.test_batch_size,
                shuffle=self.test_shuffle,
                num_workers=self.num_workers,
                drop_last=self.drop_last
            )

        if self.name in [
            "inaturalist", "colonoscopy", "spacenet"
        ]:
            # pass datapath directly to ImageFolder
            # these ones are 10,000 in size
            # 12312 for colonoscopy
            # http://www.depeca.uah.es/colonoscopy_dataset/
            self.test_only = True
            print(f"{self.name} is only available for testing/OOD detection")

            self.test_set = tv.datasets.ImageFolder(
                root=self.datapath,
                transform=self.transforms["test"],
                target_transform=self.target_transforms["test"]
            )

            self.test_loader = DataLoader(
                self.test_set,
                batch_size=self.test_batch_size,
                shuffle=self.test_shuffle,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                pin_memory=True
            )

        if self.name == "colorectal":
            # 5000 samples
            self.test_only = True
            print(f"{self.name} is only available for testing/OOD detection")

            self.test_set = tv.datasets.ImageFolder(
                root=self.datapath,
                transform=self.transforms["test"],
                target_transform=self.target_transforms["test"]
            )

            self.test_loader = DataLoader(
                self.test_set,
                batch_size=self.test_batch_size,
                shuffle=self.test_shuffle,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                pin_memory=True
            )


        # for the following two 
        # datalist.txt needs to be in the directory pointed to by datapath
        # specific samples are selected using code based on 
        # https://github.com/haoqiwang/vim/blob/master/list_dataset.py
        if self.name == "openimage-o":
            self.test_only = True
            print(f"{self.name} is only available for testing/OOD detection")

            # this only gets a subset of the images in the directory
            self.test_set = ImageFilelist(
                root=os.path.join(self.datapath, "test"),
                flist=os.path.join(self.datapath, "openimages_datalist.txt"),
                transform=self.transforms["test"],
                target_transform=self.target_transforms["test"]
            )

            self.test_loader = DataLoader(
                self.test_set,
                batch_size=self.test_batch_size,
                shuffle=self.test_shuffle,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                pin_memory=True
            )

        if self.name == "textures":
            self.test_only = True
            print(f"{self.name} is only available for testing/OOD detection")

            # this only gets a subset of the images in the directory
            self.test_set = ImageFilelist(
                root=os.path.join(self.datapath, "images"),
                flist=os.path.join(self.datapath, "textures_datalist.txt"),
                transform=self.transforms["test"],
                target_transform=self.target_transforms["test"]
            )

            self.test_loader = DataLoader(
                self.test_set,
                batch_size=self.test_batch_size,
                shuffle=self.test_shuffle,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                pin_memory=True
            )

        if "noise" in self.name:
            self.test_only = True
            print(f"{self.name} is only available for testing/OOD detection")
            # dataset size is 10000
            self.test_set = tv.datasets.ImageFolder(
                self.datapath,
                transform=self.transforms["test"],
                target_transform=self.target_transforms["test"]
            )

            self.test_loader = DataLoader(
                self.test_set,
                batch_size=self.test_batch_size,
                shuffle=self.test_shuffle,
                num_workers=self.num_workers,
                drop_last=self.drop_last
            )


# based on implementation from
# https://github.com/haoqiwang/vim/blob/master/list_dataset.py
def default_loader(path):
	return Image.open(path).convert('RGB')
def default_flist_reader(flist):
	"""
	flist format: impath label\nimpath label\n
	"""
	imlist = []
	with open(flist, 'r') as rf:
		for line in rf.readlines():
			data = line.strip().rsplit(maxsplit=1)
			if len(data) == 2:
				impath, imlabel = data
			else:
				impath, imlabel = data[0], 0
			imlist.append( (impath, int(imlabel)) )

	return imlist

class ImageFilelist(torch.utils.data.Dataset):
	def __init__(self, root, flist, transform=None, target_transform=None,
			flist_reader=default_flist_reader, loader=default_loader):
		self.root   = root
		self.imlist = flist_reader(flist)
		self.transforms = transform
		self.target_transforms = target_transform
		self.loader = loader

	def __getitem__(self, index):
		impath, target = self.imlist[index]
		img = self.loader(os.path.join(self.root,impath))
		if self.transforms is not None:
			img = self.transforms(img)
		if self.target_transforms is not None:
			target = self.target_transforms(target)

		return img, target

	def __len__(self):
		return len(self.imlist)

# just here for reference
IMAGENET200_classes = [
    'n01443537',
    'n01629819',
    'n01641577',
    'n01644900',
    'n01698640',
    'n01742172',
    'n01768244',
    'n01770393',
    'n01774384',
    'n01774750',
    'n01784675',
    'n01855672',
    'n01882714',
    'n01910747',
    'n01917289',
    'n01944390',
    'n01945685',
    'n01950731',
    'n01983481',
    'n01984695',
    'n02002724',
    'n02056570',
    'n02058221',
    'n02074367',
    'n02085620',
    'n02094433',
    'n02099601',
    'n02099712',
    'n02106662',
    'n02113799',
    'n02123045',
    'n02123394',
    'n02124075',
    'n02125311',
    'n02129165',
    'n02132136',
    'n02165456',
    'n02190166',
    'n02206856',
    'n02226429',
    'n02231487',
    'n02233338',
    'n02236044',
    'n02268443',
    'n02279972',
    'n02281406',
    'n02321529',
    'n02364673',
    'n02395406',
    'n02403003',
    'n02410509',
    'n02415577',
    'n02423022',
    'n02437312',
    'n02480495',
    'n02481823',
    'n02486410',
    'n02504458',
    'n02509815',
    'n02666196',
    'n02669723',
    'n02699494',
    'n02730930',
    'n02769748',
    'n02788148',
    'n02791270',
    'n02793495',
    'n02795169',
    'n02802426',
    'n02808440',
    'n02814533',
    'n02814860',
    'n02815834',
    'n02823428',
    'n02837789',
    'n02841315',
    'n02843684',
    'n02883205',
    'n02892201',
    'n02906734',
    'n02909870',
    'n02917067',
    'n02927161',
    'n02948072',
    'n02950826',
    'n02963159',
    'n02977058',
    'n02988304',
    'n02999410',
    'n03014705',
    'n03026506',
    'n03042490',
    'n03085013',
    'n03089624',
    'n03100240',
    'n03126707',
    'n03160309',
    'n03179701',
    'n03201208',
    'n03250847',
    'n03255030',
    'n03355925',
    'n03388043',
    'n03393912',
    'n03400231',
    'n03404251',
    'n03424325',
    'n03444034',
    'n03447447',
    'n03544143',
    'n03584254',
    'n03599486',
    'n03617480',
    'n03637318',
    'n03649909',
    'n03662601',
    'n03670208',
    'n03706229',
    'n03733131',
    'n03763968',
    'n03770439',
    'n03796401',
    'n03804744',
    'n03814639',
    'n03837869',
    'n03838899',
    'n03854065',
    'n03891332',
    'n03902125',
    'n03930313',
    'n03937543',
    'n03970156',
    'n03976657',
    'n03977966',
    'n03980874',
    'n03983396',
    'n03992509',
    'n04008634',
    'n04023962',
    'n04067472',
    'n04070727',
    'n04074963',
    'n04099969',
    'n04118538',
    'n04133789',
    'n04146614',
    'n04149813',
    'n04179913',
    'n04251144',
    'n04254777',
    'n04259630',
    'n04265275',
    'n04275548',
    'n04285008',
    'n04311004',
    'n04328186',
    'n04356056',
    'n04366367',
    'n04371430',
    'n04376876',
    'n04398044',
    'n04399382',
    'n04417672',
    'n04456115',
    'n04465501',
    'n04486054',
    'n04487081',
    'n04501370',
    'n04507155',
    'n04532106',
    'n04532670',
    'n04540053',
    'n04560804',
    'n04562935',
    'n04596742',
    'n04597913',
    'n06596364',
    'n07579787',
    'n07583066',
    'n07614500',
    'n07615774',
    'n07695742',
    'n07711569',
    'n07715103',
    'n07720875',
    'n07734744',
    'n07747607',
    'n07749582',
    'n07753592',
    'n07768694',
    'n07871810',
    'n07873807',
    'n07875152',
    'n07920052',
    'n09193705',
    'n09246464',
    'n09256479',
    'n09332890',
    'n09428293',
    'n12267677'
]
