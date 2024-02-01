# Fract'ol README
<p align="center">
  <a href="https://github.com/Dangerdrive/fract-ol">
    <img src="https://raw.githubusercontent.com/Dangerdrive/Dangerdrive/main/images/42projects/fract-olm.png" alt="Fract-ol" title="Fract-ol"/>
  </a>
</p>

This project is an implementation of the Fract'ol program using the CODAM MLX library.

## Installation

### Requirements

* C compiler
* Make

### Compilation

```bash
make


./fractol [fractal]


Where `[fractal]` is one of the following:

* `mandelbrot`
* `julia`

## Additional notes

* The included `libmlx42.a` is a pre-compiled version of the CODAM MLX library.
* The `suppress.sup` file can be used to suppress known memory leaks in the CODAM MLX library.

## Tips

* Avoid calling functions on hooks without conditions.
* Keep max iterations low, especially for testing.
* Use a lower precision data type for better performance.
* Be mindful of the performance impact of color shading.

## Bonus features

* Arrow key movement
* Color changing
* Zooming at mouse point

## Additional fractals

* Search for Mandelbrot and Julia variations.

## Extras

* Try making the fractals more aesthetically pleasing.
* Experiment with dynamically changing the Julia constant.

## Resources

* [Oceano's Fract'ol tutorial](https://www.youtube.com/watch?v=ANLW1zYbLcs)
* [Wikipedia article on plotting algorithms](https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set)
* [Fract'ol implementation with CODAM MLX](https://github.com/rvan-mee/Improved_Fractol/tree/master)
* [Another Fract'ol implementation with CODAM MLX](https://github.com/Bde-meij/fract-ol)
* [Fract'ol on GPU](https://github.com/paulo-santana/fractol-gpu)

