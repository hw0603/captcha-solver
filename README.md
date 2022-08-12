# captcha-solver
Simple captcha solver using TensorFlow.  
uses a combination of CNN and RNN models, and utilizes CTC loss to deal with unsegmented sequence data.

## Details
- `captcha2str.py`
  - Main logic of CaptchaSolver. Uses 
- `model_test.py`
  - Driver code for testing Keras models (`.h5`, `.tflite`) stored in local machine.
- `renamer.py`
  - A tool for quickly labeling sample CAPTCHA images to create training datasets.
- `webserver/server.py`
  - Implementation of test API of CaptchaSolver

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
