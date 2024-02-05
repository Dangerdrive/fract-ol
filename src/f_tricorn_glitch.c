/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   f_tricorn_glitch.c                                 :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fde-alen <fde-alen@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2023/10/07 18:50:01 by fde-alen          #+#    #+#             */
/*   Updated: 2023/10/19 18:40:58 by fde-alen         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "fractol.h"

/**
 * Initializes the fractal structure with parameters for rendering the 
 * Tricorn fractal.
 *
 * @param[in] fractal The fractal struct to be initialized.
 */
void	tricorn_data_init_glitch(t_fractal *fractal)
{
	fractal->name = "❄️ Tricorn 2 ❄️";
	fractal->color = TOMATO;
	fractal->id = TRICORN2;
	fractal->escape_value = 4.0;
	fractal->iterations = ESCAPE_COUNT;
	fractal->x_shift = -0.1;
	fractal->y_shift = 0.0;
	fractal->initial_zoom = 0.95;
	fractal->zoom = 1.0;
	fractal->xmin = -2.0 * fractal->initial_zoom;
	fractal->xmax = 2.0 * fractal->initial_zoom;
	fractal->ymin = -2.0 * fractal->initial_zoom;
	fractal->ymax = 2.0 * fractal->initial_zoom;
}

/**
 * Renders a Tricorn fractal pixel at coordinates (x, y) within the given 
 * fractal struct.
 *
 * @param[in] x       The x-coordinate of the pixel.
 * @param[in] y       The y-coordinate of the pixel.
 * @param[in] fractal The fractal structure containing rendering parameters.
 */
void	handle_tricorn_pixel_glitch(int x, int y, t_fractal *fractal)
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
		z = (complex_sum(complex_conjugate((complex_sqr(z))), c));
		if (((z.real * z.real) + (z.i * z.i)) > fractal->escape_value)
		{
			fractal->color2 = map_color(i, fractal);
			mlx_put_pixel(fractal->img, x, y, fractal->color2);
			return ;
		}
		i++;
	}
}

/**
 * Renders the Tricorn fractal on the fractal's image.
 * Glitchy version does not have pixels inside the set.
 * So when you zoom, what is inside the set won't update.
 *
 * This function iterates through all pixels in the fractal's image and calls
 * 'handle_tricorn_pixel' to compute and update the colors based on the 
 * Tricorn algorithm. It then displays the rendered fractal using 
 * the MLX library.
 *
 * @param[in,out] fractal A pointer to the fractal structure containing 
 * rendering parameters.
 */
void	tricorn_render_glitch(t_fractal *fractal)
{
	int	y;
	int	x;

	y = -1;
	while (++y < HEIGHT)
	{
		x = -1;
		while (++x < WIDTH)
			handle_tricorn_pixel_glitch(x, y, fractal);
	}
	mlx_image_to_window(fractal->mlx, fractal->img, 0, 0);
}
