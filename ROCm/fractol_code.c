/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   fractol_code.c                                     :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fde-alen <fde-alen@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2023/10/07 22:36:37 by fde-alen          #+#    #+#             */
/*   Updated: 2023/10/20 10:41:49 by fde-alen         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef FRACTOL_H
# define FRACTOL_H
# define ESCAPE_COUNT 100
# define WIDTH 720
# define HEIGHT 720
# include <fcntl.h>    // for open
# include <unistd.h>   // for close, read, write
# include <stdlib.h>   // for malloc, free, exit
# include <stdio.h>    // for perror
# include <string.h>   // for strerror
# include <math.h>
// # include "MLX42Codam/MLX42.h"
# include <hip/hip_runtime.h> // for AMD GPU (ROCm)....eventually

struct t_fractal {
  std::string name;
  int color;
  int id;
  double escape_value;
  unsigned int iterations;
  double x_shift;
  double y_shift;
  double initial_zoom;
  double zoom;
  double xmin;
  double xmax;
  double ymin;
  double ymax;
  hipDeviceptr_t img;
};

__device__ t_complex complex_sum(t_complex a, t_complex b) {
  t_complex result;
  result.real = a.real + b.real;
  result.i = a.i + b.i;
  return result;
}

__device__ t_complex complex_sqr(t_complex c) {
  t_complex result;
  result.real = c.real * c.real - c.i * c.i;
  result.i = 2 * c.real * c.i;
  return result;
}

__device__ int darken_color(t_fractal *fractal) {
  int red = (fractal->color >> 16) & 0xFF;
  int green = (fractal->color >> 8) & 0xFF;
  int blue = fractal->color & 0xFF;

  red = (int)((double)red * 0.8);
  green = (int)((double)green * 0.8);
  blue = (int)((double)blue * 0.8);

  // Combine the darkened color components
  return (red << 16) | (green << 8) | blue;
}

__device__ int map_color(unsigned int i, t_fractal *fractal) {
  int red;
  int green;
  int blue;

  red = (int)((double)i * 254 / fractal->iterations);
  green = 0;
  blue = 255 - red;

  return (red << 16) | (green << 8) | blue;
}

__global__ void handle_mandelbrot_pixel_kernel(int width, int height, t_fractal *fractal) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  t_complex z;
  t_complex c;
  unsigned int i;

  i = 0;
  z.real = 0.0;
  z.i = 0.0;
  c.real = ((fractal->xmax - fractal->xmin) * (x - 0)) / (width - 0) + fractal->xmin + fractal->x_shift;
  c.i = ((fractal->ymax - fractal->ymin) * (y - 0)) / (height - 0) + fractal->ymin + fractal->y_shift;

  while (i < fractal->iterations) {
    z = complex_sum(complex_sqr(z), c);

    if (((z.real * z.real) + (z.i * z.i)) < fractal->escape_value) {
      // Darken the color
      int color = darken_color(fractal);
      // Update the pixel color
      hipThreadStore_int(fractal->img + y * width + x, color);
    } else {
      // Map the color based on the iteration count
      int color = map_color(i, fractal);
      // Update the pixel color
      hipThreadStore_int(fractal->img + y * width + x, color);
      return;
    }

    i++;
  }
}



/**
 * @enum t_sets
 * @brief Represents different types of fractal sets.
 *
 * This enumeration defines symbolic constants representing various types
 * of fractal sets, including Mandelbrot, Julia and Tricorn.
 * These constants are used to identify and specify the type of fractal
 * being rendered or manipulated in the code.
 */
typedef enum sets
{
	MANDELBROT,
	// JULIA,
	// TRICORN,
	// MANDELBROT2,
	// JULIA2,
	// TRICORN2,
}	t_sets;

/**
 * @struct t_complex
 * @brief A complex number.
 *
 * This struct represents a complex number. It has two members:
 *
 *  @param real: The real part of the number and X axis of the complex plane.
 *  @param imaginary: The imaginary part of the number and
 * 	Y axis of the complex plane.
 */
typedef struct s_complex
{
	double		real;
	double		i;
}	t_complex;

/**
 * @struct t_fractal
 * @brief Represents a fractal rendering configuration.
 *
 * This structure stores information related to fractal rendering and
 * configuration settings. It is used to control the rendering of different
 * types of fractals and manage their properties.
 */
typedef struct s_fractal
{
	char			*name;
	int				id;
	mlx_image_t		*img;
	void			*mlx;
	double			xmin;
	double			xmax;
	double			ymin;
	double			ymax;
	double			xzoom;
	double			yzoom;
	double			zoom_factor;
	t_complex		c;
	double			escape_value;
	double			initial_zoom;	
	double			zoom;
	double			x_shift;
	double			y_shift;
	int				mouse_x;
	int				mouse_y;
	unsigned int	iterations;
	unsigned int	color;
	unsigned int	color2;
	unsigned int	r;
	unsigned int	g;
	unsigned int	b;
}				t_fractal;


/**
 * Adds two complex numbers together.
 *
 * This function takes two complex numbers 'a' and 'b' and computes their sum,
 * resulting in a new complex number 'sum'.
 *
 * @param[in] a The first complex number to be added.
 * @param[in] b The second complex number to be added.
 *
 * @return The sum of the two complex numbers 'a' and 'b'.
 */
t_complex	complex_sum(t_complex a, t_complex b)
{
	t_complex	c;

	c.real = a.real + b.real;
	c.i = a.i + b.i;
	return (c);
}

/**
 * Computes the square of a complex number.
 * * 
 * This function takes a complex number 'a' and computes its square,
 * resulting in a new complex number 'squared'.
 *
 * @param[in] a The complex number to be squared.
 *
 * @return The square of the complex number 'a'.
 */
t_complex	complex_sqr(t_complex a)
{
	t_complex	c;

	c.real = a.real * a.real - a.i * a.i;
	c.i = 2 * a.real * a.i;
	return (c);
}

/**
 * @brief Computes the complex conjugate of a complex number.
 *
 * This function takes a complex number 'a' and computes its complex conjugate,
 * resulting in a new complex number 'conjugate':
 * it mantains the real part and invert the imaginary part.
 *
 * @param[in] a The complex number for which to compute the complex conjugate.
 *
 * @return The complex conjugate of the complex number 'a'.
 */
t_complex	complex_conjugate(t_complex a)
{
	t_complex	c;

	c.real = a.real;
	c.i = -a.i;
	return (c);
}

/**
 * Computes the power of a complex number to a positive integer exponent.
 * 
 * This function raises a complex number 'a' to the power of a 
 * positive integer 'exponent',
 * resulting in a new complex number 'result'.
 *
 * @param[in] a The complex number to be raised to the power 'n'.
 * @param[in] exponent The positive integer exponent.
 *
 * @return The result of raising 'a' to the power 'n'.
 */
t_complex	complex_power(t_complex a, int exponent)
{
	t_complex	result;
	t_complex	temp;
	int			i;

	result.real = 1.0;
	result.i = 0.0;
	i = 0;
	while (i < exponent)
	{
		temp = result;
		result.real = temp.real * a.real - temp.i * a.i;
		result.i = temp.real * a.i + temp.i * a.real;
		i++;
	}
	return (result);
}


/**
 * Changes thecolor for the fractal being rendered.
 *
 * This function assigns a color value to the fractal's 'color' field.
 *
 * @param[in,out] fractal A pointer to the fractal structure to be updated 
 * with the selected color.
 */

// void	pick_color(t_fractal *fractal)
// {
// 	if (fractal->color == CYAN)
// 		fractal->color = GOLD;
// 	else if (fractal->color == GOLD)
// 		fractal->color = TOMATO;
// 	else if (fractal->color == TOMATO)
// 		fractal->color = PINK;
// 	else if (fractal->color == PINK)
// 		fractal->color = ORANGER;
// 	else if (fractal->color == ORANGER)
// 		fractal->color = VIOLET;
// 	else if (fractal->color == VIOLET)
// 		fractal->color = TEAL;
// 	else if (fractal->color == TEAL)
// 		fractal->color = BROWN;
// 	else if (fractal->color == BROWN)
// 		fractal->color = MAGENTA;
// 	else if (fractal->color == MAGENTA)
// 		fractal->color = YELLOW;
// 	else
// 		fractal->color = CYAN;
// }

/**
 * Darkens a color by a factor of 10%.
 *
 * This function takes a 32-bit color value as input and darkens it by a factor
 * of 10%. It then returns the new color value.
 *
 * @param[in] base_color The 32-bit color value to be darkened.
 *
 * @return The darkened color value.
 */
unsigned int	darken_color(t_fractal *fractal)
{	
	fractal->r = (fractal->color >> 24 & 0xFF) * 0.100;
	fractal->g = (fractal->color >> 16 & 0xFF) * 0.100;
	fractal->b = (fractal->color >> 8 & 0xFF) * 0.100;
	return ((fractal->r << 24) | (fractal->g << 16) | (fractal->b << 8) | 255);
}

/**
 * Maps an iteration count to a color for smoother gradient-based coloring.
 *
 * This function takes an iteration count, a base color, and a fractal
 * structure to compute a color based on a smoothed interpolation factor.
 * The interpolated color is calculated to create smoother gradient
 * transitions between colors in the fractal rendering.
 *
 * @param[in] iter The current iteration count.
 * @param[in] color The base color used for interpolation.
 * @param[in] fractal A pointer to the fractal structure.
 *
 * @return The interpolated color based on the iteration count.
 */
unsigned int	map_color(int iter, t_fractal *fractal)
{
	double	interpolation_factor;
	double	smoothed_factor;

	fractal->r = fractal->color >> 24 & 0xFF;
	fractal->g = fractal->color >> 16 & 0xFF;
	fractal->b = fractal->color >> 8 & 0xFF;
	interpolation_factor = (double)iter / (double)fractal->iterations;
	smoothed_factor = pow(interpolation_factor, 0.9);
	if ((interpolation_factor < smoothed_factor * 5))
	{
		fractal->r *= smoothed_factor;
		fractal->g *= smoothed_factor;
		fractal->b *= smoothed_factor;
	}
	else
	{
		// if (mlx_is_key_down(fractal->mlx, MLX_KEY_G))
		// 	pick_color(fractal);
		fractal->r *= ((smoothed_factor) * 0.9);
		fractal->g *= ((smoothed_factor) * 0.9);
		fractal->b *= ((smoothed_factor) * 0.9);
	}
	return ((fractal->r << 24) | (fractal->g << 16) | (fractal->b << 8) | 255);
}



/**
 * Initializes the Mandelbrot fractal data in the given fractal structure.
 *
 * @param[in] fractal The fractal structure to be initialized.
 */
void	mandelbrot_data_init(t_fractal *fractal)
{
	fractal->name = "❄️ Mandelbrot ❄️";
	fractal->color = BROWN;
	fractal->id = MANDELBROT;
	fractal->escape_value = 4.0;
	fractal->iterations = ESCAPE_COUNT;
	fractal->x_shift = -0.7;
	fractal->y_shift = 0.0;
	fractal->initial_zoom = 0.7;
	fractal->zoom = 1.0;
	fractal->xmin = -2.0 * fractal->initial_zoom;
	fractal->xmax = 2.0 * fractal->initial_zoom;
	fractal->ymin = -2.0 * fractal->initial_zoom;
	fractal->ymax = 2.0 * fractal->initial_zoom;
}

// /**
//  * Handles the rendering of a Mandelbrot fractal pixel.
//  *
//  * @param[in] x The x-coordinate of the pixel.
//  * @param[in] y The y-coordinate of the pixel.
//  * @param[in] fractal The fractal structure containing rendering parameters.
//  */
// void	handle_mandelbrot_pixel(int x, int y, t_fractal *fractal)
// {
// 	t_complex		z;
// 	t_complex		c;
// 	unsigned int	i;

// 	i = 0;
// 	z.real = 0.0;
// 	z.i = 0.0;
// 	c.real = ((fractal->xmax - fractal->xmin) * (x - 0))
// 		/ (WIDTH - 0) + fractal->xmin + fractal->x_shift;
// 	c.i = ((fractal->ymax - fractal->ymin) * (y - 0))
// 		/ (HEIGHT - 0) + fractal->ymin + fractal->y_shift;
// 	while (i < fractal->iterations)
// 	{
// 		z = complex_sum(complex_sqr(z), c);
// 		if ((((z.real * z.real) + (z.i * z.i)) < fractal->escape_value))
// 			mlx_put_pixel(fractal->img, x, y, darken_color(fractal));
// 		else if ((z.real * z.real + z.i * z.i) > fractal->escape_value)
// 		{
// 			fractal->color2 = map_color(i, fractal);
// 			mlx_put_pixel(fractal->img, x, y, fractal->color2);
// 			return ;
// 		}
// 		i++;
// 	}
// }

// /**
//  * Renders the Mandelbrot fractal on the fractal's image.
//  *
//  * This function iterates through all pixels in the fractal's image and calls
//  * 'handle_mandelbrot_pixel' to compute and update the colors based on the 
//  * Mandelbrot algorithm. It then displays the rendered fractal using 
//  * the MLX library.
//  *
//  * @param[in,out] fractal A pointer to the fractal structure containing 
//  * rendering parameters.
//  */
// void	mandelbrot_render(t_fractal *fractal)
// {
// 	int	y;
// 	int	x;

// 	y = -1;
// 	while (++y < HEIGHT)
// 	{
// 		x = -1;
// 		while (++x < WIDTH)
// 			handle_mandelbrot_pixel(x, y, fractal);
// 	}
// 	mlx_image_to_window(fractal->mlx, fractal->img, 0, 0);
// }

/* To convert the handle_mandelbrot_pixel function to ROCm, we need to replace the mlx_put_pixel function with a HIP equivalent. However, HIP doesn't have a direct equivalent for mlx_put_pixel because it's a high-level function for setting a pixel in an image buffer.

In HIP, you would typically use a kernel function to perform operations on the GPU. The kernel function would be launched from the host code, and it would perform the operations on the GPU.

Here's an example of how you might modify the handle_mandelbrot_pixel function to use a HIP kernel: */

__global__ void handle_mandelbrot_pixel_kernel(int x, int y, t_fractal *fractal, int *img)
{
	t_complex		z;
	t_complex		c;
	unsigned int	i;

	i = 0;
	z.real = 0.0;
	z.i = 0.0;
	c.real = ((fractal->xmax - fractal->xmin) * (x - 0))
		/ (WIDTH - 0) + fractal->xmin + fractal->x_shift;
	c.i = ((fractal->ymax - fractal->ymin) * (y - 0))
		/ (HEIGHT - 0) + fractal->ymin + fractal->y_shift;
	while (i < fractal->iterations)
	{
		z = complex_sum(complex_sqr(z), c);
		if ((((z.real * z.real) + (z.i * z.i)) < fractal->escape_value))
			img[y * WIDTH + x] = darken_color(fractal);
		else if ((z.real * z.real + z.i * z.i) > fractal->escape_value)
		{
			fractal->color2 = map_color(i, fractal);
			img[y * WIDTH + x] = fractal->color2;
			return ;
		}
		i++;
	}
}

void handle_mandelbrot_pixel(int x, int y, t_fractal *fractal)
{
	hipLaunchKernelGGL(handle_mandelbrot_pixel_kernel, dim3(1), dim3(1), 0, 0, x, y, fractal, fractal->img);
}








/**
 * Selects a fractal type and initializes its data.
 *
 * This function selects a fractal type based on the 'id' parameter of
 * the provided 'fractal' structure and initializes its data accordingly.
 * It also prints a message to inform the user about the selected fractal type.
 *
 * @param[in,out] fractal A pointer to the fractal structure to be initialized
 * and updated.
 */
void	select_fractal(t_fractal *fractal)
{
	if (fractal->id == MANDELBROT)
	{
		fractal->c.real = (drand48() * 1.2) - 0.8;
		fractal->c.i = (drand48() * 1.4) - 0.7;
		julia_data_init(fractal, fractal->c.real, fractal->c.i);
	}
	else if (fractal->id == JULIA)
		tricorn_data_init(fractal);
	else if (fractal->id == TRICORN)
		mandelbrot_data_init_glitch(fractal);
	else if (fractal->id == MANDELBROT2)
	{
		fractal->c.real = (drand48() * 1.2) - 0.8;
		fractal->c.i = (drand48() * 1.4) - 0.7;
		julia_data_init(fractal, fractal->c.real, fractal->c.i);
	}
	else if (fractal->id == JULIA2)
		tricorn_data_init_glitch(fractal);
	else
		mandelbrot_data_init(fractal);
	mlx_set_window_title(fractal->mlx, fractal->name);
}

/**
 * Initializes a fractal structure with the specified fractal type and
 * parameters.
 *
 * This function initializes a 'fractal' structure with the provided 'id'
 * parameter representing the fractal type. If 'id' corresponds to 
 * JULIA set, additional parameters 'c_x' and 'c_y' are used.
 *
 * @param[out] fractal A pointer to the fractal structure to be initialized.
 * @param[in] id The fractal type identifier (MANDELBROT, JULIA, etc.).
 * @param[in] c_x The x-coordinate for Julia set initialization.
 * @param[in] c_y The y-coordinate for Julia set initialization.
 */
// void	fractal_init(t_fractal *fractal, int id, double c_x, double c_y)
// {
// 	if (id == MANDELBROT)
// 		mandelbrot_data_init(fractal);
// 	// if (id == JULIA)
// 	// 	julia_data_init(fractal, c_x, c_y);
// 	// if (id == TRICORN)
// 	// 	tricorn_data_init(fractal);
// 	// guide();

// 	//------converts mallocs to ROCm	
// 	// fractal->mlx = mlx_init(WIDTH, HEIGHT, fractal->name, false);
// 	// if (!fractal->mlx)
// 	// 	exit(EXIT_FAILURE);
// 	// fractal->img = mlx_new_image(fractal->mlx, WIDTH, HEIGHT);
// 	// if (!fractal->img)
// 	// {
// 	// 	mlx_terminate(fractal->mlx);
// 	// 	exit(EXIT_FAILURE);
// 	// }
// }
void	fractal_init(t_fractal *fractal, int id, double c_x, double c_y)
{
	if (id == MANDELBROT)
		mandelbrot_data_init(fractal);

	hipMalloc((void**)&fractal->mlx, WIDTH * HEIGHT * sizeof(int));
	if (!fractal->mlx)
		exit(EXIT_FAILURE);
	hipMalloc((void**)&fractal->img, WIDTH * HEIGHT * sizeof(int));
	if (!fractal->img)
	{
		hipFree(fractal->mlx);
		exit(EXIT_FAILURE);
	}
}



/**
 * @brief Update the rendering of the fractal based on its current type.
 *
 * This function updates the rendering of the fractal based on the fractal type
 * specified in the `t_fractal` structure. It calls the appropriate rendering
 * function for the selected fractal type.
 *
 * @param[in, out] fractal A pointer to the `t_fractal` structure containing
 * the fractal type and rendering parameters.
 */
void	update_render(t_fractal *fractal)
{
	if (fractal->id == MANDELBROT)
	{
		mandelbrot_render(fractal);
	}
	// if (fractal->id == JULIA)
	// {
	// 	julia_render(fractal);
	// }
	// if (fractal->id == TRICORN)
	// {
	// 	tricorn_render(fractal);
	// }
	// if (fractal->id == MANDELBROT2)
	// {
	// 	mandelbrot_render_glitch(fractal);
	// }
	// if (fractal->id == JULIA2)
	// {
	// 	julia_render_glitch(fractal);
	// }
	// if (fractal->id == TRICORN2)
	// {
	// 	tricorn_render_glitch(fractal);
	// }
}


/**
 * @brief Function to check the input and initialize the fractal.
 *
 * @param data The data struct
 */
// static bool	check_input(int argc, char **argv, t_fractal *fractal)
// {
// 	if (argc == 2 && !strncmp(argv[1], "mandelbrot", 10))
// 		fractal_init(fractal, MANDELBROT, 0, 0);
// 	else if (argc == 4 && !strncmp(argv[1], "julia", 5))
// 	{
// 		if (ft_atod(argv[2]) >= -2.0 && ft_atod(argv[2]) <= 2.0
// 			&& ft_atod(argv[3]) >= -2.0 && ft_atod(argv[3]) <= 2.0)
// 			fractal_init(fractal, JULIA, ft_atod(argv[2]), ft_atod(argv[3]));
// 		else
// 			return (false);
// 	}
// 	else if (argc == 2 && !strncmp(argv[1], "tricorn", 7))
// 		fractal_init(fractal, TRICORN, 0, 0);
// 	else
// 		return (false);
// 	return (true);
// }

// int	main(int argc, char **argv)
// {
// 	t_fractal		fractal;

// 	fractal_init(fractal);



// 	update_render(&fractal);

// 		//-------transform hooks to ROCm
// 		// mlx_loop_hook(fractal.mlx, keyhook, &fractal);
// 		// mlx_scroll_hook(fractal.mlx, &scrollhook, &fractal);
// 		// mlx_cursor_hook(fractal.mlx, &cursorhook, &fractal);
// 		// mlx_loop(fractal.mlx);
	
// 	//------------transform to free ROCm mallocs
// 	// mlx_delete_image(fractal.mlx, fractal.img);
// 	// mlx_terminate(fractal.mlx);
// 	return (EXIT_SUCCESS);
// }
int	main(int argc, char **argv)
{
	t_fractal		fractal;

	fractal_init(&fractal);

	update_render(&fractal);

	hipStreamCreate(&fractal.mlx);
	hipStreamSynchronize(fractal.mlx);
	hipStreamDestroy(fractal.mlx);

	hipFree(fractal.mlx);
	hipFree(fractal.img);
	hipDeviceReset();
	return (EXIT_SUCCESS);
}
