/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   f_julia_glitch.c                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fde-alen <fde-alen@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2023/10/07 18:49:54 by fde-alen          #+#    #+#             */
/*   Updated: 2023/10/19 18:33:33 by fde-alen         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "fractol.h"

/**
 * Initializes the fractal structure with Julia set parameters.
 *
 * This function initializes a 'fractal' structure with the parameters
 * specific to the Julia set. It sets the fractal's name, type, escape value,
 * maximum iterations, shifts, zoom level, and Julia constant 'c'.
 *
 * @param[in,out] fractal A pointer to the fractal structure to be initialized.
 * @param[in] c_x The real part of the Julia constant 'c'.
 * @param[in] c_y The imaginary part of the Julia constant 'c'.
 */
void	julia_data_init_glitch(t_fractal *fractal, double c_x, double c_y)
{
	fractal->name = "❄️ Julia 2 ❄️";
	fractal->id = JULIA2;
	fractal->color = CYAN;
	fractal->escape_value = 4.0;
	fractal->iterations = ESCAPE_COUNT;
	fractal->x_shift = 0.0;
	fractal->y_shift = 0.0;
	fractal->initial_zoom = 1;
	fractal->zoom = 1.0;
	fractal->c.real = c_x;
	fractal->c.i = c_y;
	fractal->xmin = -2.0 * fractal->initial_zoom;
	fractal->xmax = 2.0 * fractal->initial_zoom;
	fractal->ymin = -2.0 * fractal->initial_zoom;
	fractal->ymax = 2.0 * fractal->initial_zoom;
}

/**
 * Handles the rendering of a Julia set pixel at the specified coordinates.
 *
 * This function computes the color of a Julia set pixel at coordinates (x, y)
 * and updates the pixel in the fractal's image. It iteratively calculates
 * the color based on the Julia set algorithm until it reaches the escape
 * value or maximum iterations.
 *
 * @param[in] x The x-coordinate of the pixel.
 * @param[in] y The y-coordinate of the pixel.
 * @param[in,out] fractal A pointer to the fractal structure containing 
 * parameters and image data.
 */
void	handle_julia_pixel_glitch(int x, int y, t_fractal *fractal)
{
	t_complex		z;
	unsigned int	i;

	i = 0;
	z.real = ((x - 0) * (fractal->xmax - fractal->xmin))
		/ (WIDTH - 0) + fractal->xmin + fractal->x_shift;
	z.i = ((y - 0) * (fractal->ymax - fractal->ymin))
		/ (HEIGHT - 0) + fractal->ymin + fractal->y_shift;
	while (i < fractal->iterations)
	{
		z = complex_sum(complex_sqr(z), fractal->c);
		if ((((z.real * z.real) + (z.i * z.i)) < fractal->escape_value))
			mlx_put_pixel(fractal->img, x, y, fractal->color * 0.9995);
		else if ((z.real * z.real + z.i * z.i) > fractal->escape_value)
		{
			fractal->color2 = map_color(i, fractal);
			mlx_put_pixel(fractal->img, x, y, fractal->color2);
			return ;
		}
		i++;
	}
}

/**
 * Randomizes the Julia set constant 'c' with new values.
 *
 * This function generates random real and imaginary parts for the Julia set
 * constant 'c' within specified ranges and updates the fractal accordingly.
 *
 * @param[in,out] fractal_ptr A pointer to the fractal structure to be updated.
 */
void	randomize_julia_glitch(t_fractal *fractal_ptr)
{
	fractal_ptr->c.real = (drand48() * 1.2) - 0.8;
	fractal_ptr->c.i = (drand48() * 1.4) - 0.7;
}

/**
 * Renders the Julia fractal on the fractal's image.
 *
 * This function iterates through all pixels in the fractal's image and calls
 * 'handle_julia_pixel' to compute and update the colors based on the Julia set
 * algorithm. It then displays the rendered fractal using the MLX library.
 *
 * @param[in,out] fractal A pointer to the fractal structure containing 
 * rendering parameters.
 */
void	julia_render_glitch(t_fractal *fractal)
{
	int	y;
	int	x;

	y = -1;
	while (++y < HEIGHT)
	{
		x = -1;
		while (++x < WIDTH)
			handle_julia_pixel_glitch(x, y, fractal);
	}
	mlx_image_to_window(fractal->mlx, fractal->img, 0, 0);
}
