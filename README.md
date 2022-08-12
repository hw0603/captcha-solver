# captcha-solver
Simple captcha solver using TensorFlow.  
uses a combination of CNN and RNN models, and utilizes CTC loss to deal with unsegmented sequence data.

## Details
- `captcha2str.py`
  - Main logic of CaptchaSolver. Selectively uses TensorFlow/TFLite depending on system
- `model_test.py`
  - Driver code for testing Keras models(`.h5`, `.tflite`) stored in local machine.
- `renamer.py`
  - A tool for quickly labeling sample CAPTCHA images to create training datasets.
- `webserver/server.py`
  - Implementation of test API of CaptchaSolver

## Using CaptchaSolver with TFLite 
Tensorflow is a framework that is too heavy to run on low-power devices like Raspberry Pi, so if you use CaptchaSolver just for inference, you can use Tenserflow Lite (TFLite).  
To apply TFLite, install TFLite using the command `pip install tflite_runtime`, and then save the `.tflite` file converted from the `.h5` to the same path as the `captcha2str.py`.
  
TensorFlow Lite builtin operator library only supports a limited number of TensorFlow operators. Since CaptchaSolver also uses operators that are not builtin operators, you will need to enable the usage of `Custom OP`. (see https://www.tensorflow.org/lite/guide/ops_select)  
  
Thanks to @PINTO0309, you can find pre-built python wheel with `Custom OP` enabled for arm devices.  
Rather than installing official `tflite_runtime`, install the proper `.whl` with command `pip install ${your_whl_file_name}`

## Screenshots
<!-- ![Captcha Solver API](https://user-images.githubusercontent.com/31981462/184404660-5668718b-ac5e-443a-b7dc-2f19962902eb.png)
![Captcha Solver](https://user-images.githubusercontent.com/31981462/184404705-9caa9f98-7deb-4e49-a573-698296b491e3.png) -->
![visualization](https://user-images.githubusercontent.com/31981462/184406293-058dce58-65b7-4a6d-8f66-bb3aab7bface.PNG)
<table>
    <th>Captcha Solver</th>
    <th>Captcha Solver API</th>
    <tr>
	    <td>
            <p align="center">
                <img src="https://user-images.githubusercontent.com/31981462/184404705-9caa9f98-7deb-4e49-a573-698296b491e3.png" height="100%" width="100%">
            </p>
        </td>
        <td>
            <p align="center">
                <img src="https://user-images.githubusercontent.com/31981462/184404660-5668718b-ac5e-443a-b7dc-2f19962902eb.png" height="100%" width="100%">
            </p>
        </td>
</table>
