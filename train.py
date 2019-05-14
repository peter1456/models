from models.utils import load_data
from models.shuffle_unit import ShuffleNet
from models.mobilenet import MobileNet

import numpy as np

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam


train_data, test_data = load_data()
x_train, y_train = train_data
x_test, y_test = test_data

print(np.shape(x_train[0]))
print(np.squeeze(np.shape(y_train[0])))

n_epochs = 2


# shuffle_net = ShuffleNet(
#         # input_shape,
#         np.shape(x_train[0]),
#         # nb_classes,
#         np.squeeze(np.shape((y_train[0]))),
#         include_top=True,
#         weights=None,
#         nb_groups=8     # can vary from 1 2 3 4 8
# )
# shuffle_net.compile(
#         loss=categorical_crossentropy,
#         optimizer=Adam(),
#         metrics=['accuracy']
# )
# shuffle_net.fit(
#         x_train,
#         y_train,
#         batch_size=200,
#         epochs=n_epochs,
#         verbose=1,
#         validation_data=(x_test, y_test)
# )

mobile_net = MobileNet(
        np.shape(x_train[0]),
        np.squeeze(np.shape((y_train[0]))),
)
mobile_net.compile(
        loss=categorical_crossentropy,
        optimizer=Adam(),
        metrics=['accuracy']
)
mobile = mobile_net.fit(
        x_train,
        y_train,
        batch_size=200,
        epochs=n_epochs,
        verbose=1,
        validation_data=(x_test, y_test)
)

plot(mobile.history)

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

