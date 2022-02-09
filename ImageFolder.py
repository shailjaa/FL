import tempfile
from PIL import Image
import torch.utils.data as data

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, bucket, transform=None, target_transform=None, loader=default_loader):
        self.root = root
        self.imlist = flist
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.bucket = bucket

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        obj = self.bucket.Object(impath)
        tmp = tempfile.NamedTemporaryFile()
        tmp_name = '{}.jpg'.format(tmp.name)

        with open(tmp_name, 'wb') as f:
            obj.download_fileobj(f)
            f.flush()
            f.close()
            image = Image.open(tmp_name)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.imlist)
