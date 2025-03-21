import numpy as np
import pandas as pd
import tifffile
import tensorflow as tf
from tqdm import tqdm

model = tf.keras.models.load_model("/Users/lollijagladiseva/Desktop/single_sample_v3_75epoch_64-64-21_multiclass.keras")

def compute_gradients(inputs, model, top_pred_idx):
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        preds = model(inputs)
        class_output = preds[:, top_pred_idx]
    gradients = tape.gradient(class_output, inputs)
    return gradients

def integrated_gradients(input_image, model, top_pred_idx, baseline=None, steps=50):
    if baseline is None:
        baseline = np.zeros_like(input_image)
    alphas = np.linspace(0, 1, steps)
    interpolated_images = [baseline + alpha * (input_image - baseline) for alpha in alphas]
    
    gradients = []
    for img in tqdm(interpolated_images, desc="Computing Gradients", unit="image"):
        img = np.expand_dims(img, axis=0)
        grads = compute_gradients(img, model, top_pred_idx)
        gradients.append(grads)
    
    gradients = np.array(gradients)
    avg_gradients = (gradients[:-1] + gradients[1:]) / 2.0
    integrated_grads = np.mean(avg_gradients, axis=0) * (input_image - baseline)
    return integrated_grads[0]

results = []
balanced_samples = pd.read_csv('/Users/lollijagladiseva/Desktop/balanced_samples.csv', index_col=0)


for barcode in tqdm(balanced_samples.index, desc="Processing Samples", unit="sample"):
    X = balanced_samples.loc[barcode, 'X']
    Y = balanced_samples.loc[barcode, 'Y']
    
    input_file = "/Users/lollijagladiseva/Desktop/rotated.tiff"
    z_crop_s, z_crop_e = 296 - 10, 296 + 11
    
    with tifffile.TiffFile(input_file) as tif:
        y_crop_s, y_crop_e = int(Y) - 64, int(Y) + 64
        x_crop_s, x_crop_e = int(X) - 64, int(X) + 64
        
        cropped_slices = []
        for z in range(z_crop_s, z_crop_e):
            slice_data = tif.pages[z].asarray()
            cropped_slice = slice_data[y_crop_s:y_crop_e, x_crop_s:x_crop_e]
            cropped_slices.append(cropped_slice)

    cropped_slices_3d = np.expand_dims(np.stack(cropped_slices, axis=0), axis=-1)
    cropped_slices_3d = cropped_slices_3d.astype(float) / 255.0
    cropped_slices_3d = np.expand_dims(cropped_slices_3d, axis=-1)
    class_prediction = model.predict(np.expand_dims(cropped_slices_3d, axis=0))
    top_pred_idx = np.argmax(class_prediction[0]) 
    attributions = integrated_gradients(cropped_slices_3d, model, top_pred_idx)

    attributions_log = np.log(np.abs(attributions) + 1e-10)
    attributions_flat = attributions_log.flatten()

    results.append({
        'barcode': barcode,
        'class_idx': top_pred_idx,
        'X': X,
        'Y': Y,
        'attributions': attributions_flat
    })

df_attributions = pd.DataFrame(results)
df_attributions.to_csv('/Users/lollijagladiseva/Desktop/df_attributions.csv', index=True)
df_attributions.to_pickle('/Users/lollijagladiseva/Desktop/df_attributions.pkl')
