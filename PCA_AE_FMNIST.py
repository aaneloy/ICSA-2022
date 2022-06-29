
from libraries import *

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train.shape


x_train = x_train.reshape(x_train.shape[0], 28*28)
x_test = x_test.reshape(x_test.shape[0], 28*28)
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

m = Sequential()
m.add(Dense(1000,  activation='relu', input_shape=(784,)))
m.add(Dense(500,  activation='relu'))
m.add(Dense(250,  activation='relu'))
m.add(Dense(32,  activation='relu'))
m.add(Dense(2,    activation='linear', name="bottleneck"))
m.add(Dense(32,  activation='relu'))
m.add(Dense(250,  activation='relu'))
m.add(Dense(500,  activation='relu'))
m.add(Dense(1000,  activation='relu'))
m.add(Dense(784,  activation='sigmoid'))

m.compile(loss='mse',optimizer='adam')

history = m.fit(x_train, x_train, batch_size=128, epochs=25, verbose=1, validation_data=(x_test, x_test))

encoder = Model(m.input, m.get_layer('bottleneck').output)
encoded = encoder.predict(x_train)
decoded = m.predict(x_train)
encoder.summary()

pca=PCA(n_components=2)
x_train_pca=pca.fit_transform(x_train)
x_pca_inv=pca.inverse_transform(x_train_pca[:,:2])
loss=((x_train - x_pca_inv) ** 2).mean()
print("Reconstruction loss from PCA:",loss)

plt.figure(figsize=(7,7))
plt.plot(history.history['val_loss'], label='Val set loss')
plt.plot(history.history['loss'], label='Train set loss')
plt.hlines(y=loss, xmin=0, xmax=25, label='PCA loss', linestyles='dashed')
plt.legend(bbox_to_anchor=(1.3, 1));
plt.show()

fig, ax= plt.subplots(1,2, figsize=(12,8))
ax[0].scatter(encoded[:9000,0], encoded[:9000,1], c=y_train[:9000])
ax[0].set_title("Autoencoder")
ax[0].axis('off')

ax[1].scatter(x_train_pca[:,:2][:9000,0], x_train_pca[:,:2][:9000,1], c=y_train[:9000])
ax[1].set_title("PCA")
ax[1].axis('off');

fig = plt.figure()
fig.tight_layout()
fig.set_figwidth(15)
fig.set_figheight(10)
plot_this=[x_train, decoded, x_pca_inv]
fig.suptitle("Original-Autoencoder-PCA", fontsize=26)
for j in range(len(plot_this)):
    for i in range(10):
        fig.add_subplot(3,10,i+1+(10*j))
        plt.axis('off')
        plt.imshow(plot_this[j][i].reshape(28,28), cmap='Greys')

