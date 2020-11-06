## F0-AUTOVC: F0-Consistent Many-to-Many Non-Parallel Voice Conversion via Conditional Autoencoder  
This repository provides a PyTorch implementation of the paper [F0-AUTOVC](https://arxiv.org/abs/2004.07370).

Based on
- https://github.com/auspicious3000/autovc
- https://github.com/auspicious3000/SpeechSplit
- https://github.com/christopher-beckham/amr

## Dependencies
- Python 3.7
- Pytorch 1.6.0
- TensorFlow
- Numpy
- librosa
- tqdm

## Usage
1. Prepare dataset<br>
    we used the [VCTK dataset](http://www.udialogue.org/download/cstr-vctk-corpus.html) as used in original paper.  
    But, you can use your own dataset.
    
2. Prepare the speaker to gender file as shown in nikl_spk.txt and run ```make_spk2gen.py```  
    * Format  
    speaker1 gender1  
    speaker2 gender2  
    
    * Example:  
        p225    W  
        p226    M  
        p301    W  
        p302    W  
        .  
        .

3. Preprocess data using ```preprocess.py```  

4. Run ```task_launcher.py```