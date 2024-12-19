import tensorflow as tf
from tensorflow.keras import layers, models

def unet_model(input_size=(128, 128, 3)):
    inputs = layers.Input(input_size)
    
    # encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs) #padding='same' means no padding
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)
    
    # bottom
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# example
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data(num_samples, img_size):
    X = np.zeros((num_samples, img_size, img_size, 3), dtype=np.float32)
    y = np.zeros((num_samples, img_size, img_size, 1), dtype=np.float32)
    
    for i in range(num_samples):
        # generate random circles
        radius = np.random.randint(10, img_size//4)
        center_x = np.random.randint(radius, img_size - radius)
        center_y = np.random.randint(radius, img_size - radius)
        
        Y, X_grid = np.ogrid[:img_size, :img_size]
        dist_from_center = np.sqrt((X_grid - center_x)**2 + (Y - center_y)**2)
        mask = dist_from_center <= radius
        
        y[i, mask, 0] = 1.0
        # random color
        X[i] = np.random.rand(img_size, img_size, 3) * mask[..., np.newaxis]
    
    return X, y

# generate training and testing dataset
train_X, train_y = generate_synthetic_data(1000, 128)
val_X, val_y = generate_synthetic_data(200, 128)

# visualize some results
def display_sample(X, y, index):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.imshow(X[index])
    plt.title('Image')
    plt.subplot(1,2,2)
    plt.imshow(y[index].squeeze(), cmap='gray')
    plt.title('Mask')
    plt.show()

display_sample(train_X, train_y, 0)


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# use callback function to save the best model
checkpoint = ModelCheckpoint(filepath='unet_best_model.keras',monitor='val_loss', verbose=1, save_best_only=True, mode='min')

lr_reducer = ReduceLROnPlateau(factor=0.1,cooldown=0,patience=50,min_lr=0.5e-6,monitor='val_loss',mode = 'min',verbose= 1)
# model.fit(Xnoisy,Xnoisy,batch_size=128,verbose=1,epochs=20,callbacks=[checkpoint,lr_reducer],validation_split=0.2)

# training
history = model.fit(train_X, train_y, 
                    validation_data=(val_X, val_y),
                    epochs=20, 
                    batch_size=16,
                    callbacks=[checkpoint,lr_reducer])

# loading the best model
model.load_weights('unet_best_model.keras')

# testing 
loss, accuracy = model.evaluate(val_X, val_y)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')

# visualize results
def predict_and_display(model, X, y, index):
    pred = model.predict(X[index:index+1])[0]
    pred_mask = pred > 0.5
    
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(X[index])
    plt.title('Image')
    plt.subplot(1,3,2)
    plt.imshow(y[index].squeeze(), cmap='gray')
    plt.title('True Mask')
    plt.subplot(1,3,3)
    plt.imshow(pred_mask.squeeze(), cmap='gray')
    plt.title('Predicted Mask')
    plt.show()
predict_and_display(model, val_X, val_y, 0)

