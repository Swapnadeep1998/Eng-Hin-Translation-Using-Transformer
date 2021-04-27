import tensorflow as tf
from Src import config
from Src.model import create_model
from Src.Input.data import DataLoader

train_data_path=config.TRAIN_DATA_PATH
valid_data_path=config.VAL_DATA_PATH
batch_size=config.BATCH_SIZE
epochs=config.EPOCHS
model_path=config.MODEL_PATH


train_dataloader = DataLoader(train_data_path)
val_dataloader = DataLoader(valid_data_path)

train_data = train_dataloader.get_data(shuffle=True, batch_size=batch_size)
val_data = val_dataloader.get_data(shuffle=False, batch_size=batch_size)

model = create_model()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(optimizer=optimizer, loss=[loss, loss])


if __name__ == '__main__':

    callback = tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss")


    history = model.fit(
    train_data, validation_data = val_data,
    epochs=epochs,
    verbose = 1,
    callbacks=[callback]
)



