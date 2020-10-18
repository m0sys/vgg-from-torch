<h1 align="center">Welcome to vgg-from-torch 👋</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-1.0-blue.svg?cacheSeconds=2592000" />
  <a href="https://opensource.org/licenses/MIT" target="_blank">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
</p>

> Implementation of VGG networks using pytorch as the base.



## Install

```sh
conda env create -f environment.yml
```



## Results

- Reached 93% validation accuracy on Cifar10 dataset using Vgg11.
- Dropout helped increase validation accuracy by 3% (from approx 90% to 93%).
- Reached Similar result on Cifar10 dataset using Vgg13.
- Deeper models do not converge to a good loss.

See [assets](https://github.com/mhd53/vgg-from-torch/tree/main/_assets) for graphs.

## TODO

- [ ] Fix weight decay out of memory error and train larger models.



## Author

* Website: [morealfit](morealfit.com)
* Github: [@mhd53](https://github.com/mhd53)



## Show your support

Give a ⭐️ if this project helped you!



## 📝 License

This project is [MIT](https://opensource.org/licenses/MIT) licensed.



## 📃 Reference Papers:

- [Very Deep Convolutional Networks For Large-Scale Image Recognition, Simonyan & Zisserman](https://arxiv.org/pdf/1409.1556.pdf)

***
