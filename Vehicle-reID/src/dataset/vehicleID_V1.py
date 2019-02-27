from dataset import Dataset

ROOT='/root/shaohaolu/vehicleReid/data/VehicleID_V1.0'

class VehicleID_V1(Dataset):
       
    def __init__(self,test_size,root=ROOT):
        super(VehicleID_V1, self).__init__(root, test_size)
        
        self.load(root)
