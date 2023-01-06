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
This is the linke of the main project which I'm working on and explaining about it :https://github.com/hrnoh/f0-autovc
The following links are articles that will help you understand the subject better
[2002.00198] Transforming Spectrum and Prosody for Emotional Voice Conversion with Non-Parallel Training Data (arxiv.org)
Blow: a single-scale hyperconditioned flow for non-parallel raw-audio voice conversion (neurips.cc)
[1907.10185] Non-Parallel Voice Conversion with Cyclic Variational Autoencoder (arxiv.org)
The link below contains a video where I explain the summary of the article:https://drive.google.com/file/d/1QUo5kBkf8QxXhK-y53c4CvqTs0YDxM6z/view?usp=drivesdk
[2108.04395] StarGAN-VC+ASR: StarGAN-based Non-Parallel Voice Conversion Regularized by Automatic Speech Recognition (arxiv.org)
The last link contains my explanation about the main project:https://drive.google.com/file/d/1YxCwQQt4UT36T5Hz4iYW0nKdxEijgDy0/view?usp=drivesdk
