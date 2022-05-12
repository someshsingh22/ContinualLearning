#Imports
import learn2learn as l2l
import random
import argparse
#Global Variables


#Functions


#Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="./data/")
args = parser.parse_args()

#Main
if __name__ == "__main__":
    #Import AGNews Dataset
    dataset = l2l.text.datasets.NewsClassification(root=args.data, download=True)
    dataset = l2l.data.MetaDataset(dataset)
    classes = list(range(len(dataset.labels))) # 41 classes
    random.shuffle(classes)
    
    #Implemetnt Continual Transforms

    train_dataset, validation_dataset, test_dataset = dataset, dataset, dataset

    train_gen = l2l.data.TaskDataset(
            train_dataset, num_tasks=20000, 
            task_transforms=[
                l2l.data.transforms.FusedNWaysKShots(
                    train_dataset, n=args.ways, k=shots, filter_labels=classes[:20]),
                l2l.data.transforms.LoadData(train_dataset),
                l2l.data.transforms.RemapLabels(train_dataset)],)

    validation_gen = l2l.data.TaskDataset(
            validation_dataset, num_tasks=20000, 
            task_transforms=[
                l2l.data.transforms.FusedNWaysKShots(
                    validation_dataset, n=args.ways, k=shots, filter_labels=classes[20:30]),
                l2l.data.transforms.LoadData(validation_dataset),
                l2l.data.transforms.RemapLabels(validation_dataset)],)

    test_gen = l2l.data.TaskDataset(
            test_dataset, num_tasks=20000, 
            task_transforms=[
                l2l.data.transforms.FusedNWaysKShots(
                    test_dataset, n=args.ways, k=shots, filter_labels=classes[30:]),
                l2l.data.transforms.LoadData(test_dataset),
                l2l.data.transforms.RemapLabels(test_dataset)],)
    
    #Implement Model


    #Implemnet Training Loop
    
    pass