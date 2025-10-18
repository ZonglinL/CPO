from utils import image_grid, get_reward_model, get_reward_loss, label_transform, group_random_crop 

for task in ['depth','hed','canny','lineart']:
    get_reward_model(task)