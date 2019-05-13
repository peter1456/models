from shufflenet.io_utils import load_data
from shufflenet.shuffle_unit import ShuffleNet

import numpy as np


from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam



train_data, test_data = load_data()
x_train, y_train = train_data
x_test, y_test = test_data

print(np.shape(x_train[0]))
print(np.squeeze(np.shape(y_train[0])))

n_epochs = 2


model = ShuffleNet(
        # input_shape,
        np.shape(x_train[0]),
        # nb_classes,
        np.squeeze(np.shape((y_train[0]))),
        include_top=True,
        weights=None,
        nb_groups=8     # can vary from 1 2 3 4 8
)
model.compile(
        loss=categorical_crossentropy,
        optimizer=Adam(),
        metrics=['accuracy']
)
model.fit(
        x_train,
        y_train,
        batch_size=100,
        epochs=n_epochs,
        verbose=1,
        validation_data=(x_test, y_test)
)

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

