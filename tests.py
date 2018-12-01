import numpy as np
import train_variants

a = np.arange(15)
print('a =', a)
print()

# ---------------------------------------
print('train_variants.create_sets(4, a, a)')
imgs, labs = train_variants.create_sets(4, np.copy(a), np.copy(a))
print(imgs)

assert len(imgs) == 4
assert len(labs) == 4
assert all([np.array_equal(i, l) for i, l in zip(imgs, labs)])

print()

# ---------------------------------------
print('train_variants.get_rotations(4, imgs, labs)')
train, test = train_variants.get_rotations(4, imgs, labs)

print(train)

assert len(train) == 4
assert len(test) == 4
assert all([len(t) == 2 and (t[0].shape[0] == 12 or t[0].shape[0] == 11) and (t[1].shape[0] == 12 or t[1].shape[0] == 11) for t in train])
assert all([len(t) == 2 and (t[0].shape[0] == 4 or t[0].shape[0] == 3) and (t[1].shape[0] == 4 or t[1].shape[0] == 3) for t in test])
for i in range(4):
    assert np.intersect1d(train[i][0], test[i][0]).shape[0] == 0 and np.intersect1d(test[i][0], train[i][0]).shape[0] == 0
    assert np.union1d(train[i][0], test[i][0]).shape[0] == 15

print()
print('Success!')