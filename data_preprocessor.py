import numpy as np

def read_rating(path, num_users,num_items, a, b, upl):
    if 'ml-100k' in path:
        fp = open(path + "u.data")
        splitter = '\t'
    else:
        fp = open(path + "ratings.dat")
        splitter = '::'
    user_train_set = set()
    user_test_set = set()
    item_train_set = set()
    item_test_set = set()

    R = np.zeros((num_users,num_items))
    mask_R = np.zeros((num_users, num_items))
    C = np.ones((num_users, num_items)) * b

    train_R = np.zeros((num_users, num_items))
    test_R = np.zeros((num_users, num_items))

    train_mask_R = np.zeros((num_users, num_items))
    test_mask_R = np.zeros((num_users, num_items))

    lines = fp.readlines()
    train_idx , test_idx = _train_test_upl_split(lines, num_users, upl, splitter )

    num_train_ratings = len(train_idx)
    num_test_ratings = len(test_idx)


    for line in lines:
        user,item,rating,_ = line.split(splitter)
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        R[user_idx,item_idx] = int(rating)
        mask_R[user_idx,item_idx] = 1
        C[user_idx,item_idx] = a

    ''' Train '''
    for itr in train_idx:
        line = lines[itr]
        user,item,rating,_ = line.split(splitter)
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        train_R[user_idx,item_idx] = int(rating)
        train_mask_R[user_idx,item_idx] = 1

        user_train_set.add(user_idx)
        item_train_set.add(item_idx)

    ''' Test '''
    for itr in test_idx:
        line = lines[itr]
        user, item, rating, _ = line.split(splitter)
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        test_R[user_idx, item_idx] = int(rating)
        test_mask_R[user_idx, item_idx] = 1

        user_test_set.add(user_idx)
        item_test_set.add(item_idx)

    return R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,\
user_train_set,item_train_set,user_test_set,item_test_set


def _train_test_upl_split(data, num_users, UPL, splitter):
    train_indices = []
    test_indices = []

    user_rates_count = np.zeros(num_users)
    for row in data:
        user,item,rating,_ = row.split(splitter)
        user_rates_count[int(user)-1] +=1
    valid_users = np.where(user_rates_count  > UPL + 10)[0]

    user_included_movies = np.zeros(num_users)
    idx = 0
    for row in data:
        user,item,rating,_ = row.split(splitter)
        if int(user)-1 in valid_users:
            if user_included_movies[int(user)-1] < UPL:
                train_indices.append(idx)
                user_included_movies[int(user)-1] += 1
            else:
                test_indices.append(idx)
        idx +=1

    return train_indices, test_indices
