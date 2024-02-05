/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   fractal_init.c                                     :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fde-alen <fde-alen@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2023/10/07 22:36:37 by fde-alen          #+#    #+#             */
/*   Updated: 2023/10/20 11:37:50 by fde-alen         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "fractol.h"

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
void	fractal_init(t_fractal *fractal, int id, double c_x, double c_y)
{
	if (id == MANDELBROT)
		mandelbrot_data_init(fractal);
	if (id == JULIA)
		julia_data_init(fractal, c_x, c_y);
	if (id == TRICORN)
		tricorn_data_init(fractal);
	guide();
	fractal->mlx = mlx_init(WIDTH, HEIGHT, fractal->name, false);
	if (!fractal->mlx)
		exit(EXIT_FAILURE);
	fractal->img = mlx_new_image(fractal->mlx, WIDTH, HEIGHT);
	if (!fractal->img)
	{
		mlx_terminate(fractal->mlx);
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
	if (fractal->id == JULIA)
	{
		julia_render(fractal);
	}
	if (fractal->id == TRICORN)
	{
		tricorn_render(fractal);
	}
	if (fractal->id == MANDELBROT2)
	{
		mandelbrot_render_glitch(fractal);
	}
	if (fractal->id == JULIA2)
	{
		julia_render_glitch(fractal);
	}
	if (fractal->id == TRICORN2)
	{
		tricorn_render_glitch(fractal);
	}
}
