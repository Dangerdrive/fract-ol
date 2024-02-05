/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   colors.c                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fde-alen <fde-alen@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2023/10/12 17:41:15 by fde-alen          #+#    #+#             */
/*   Updated: 2023/10/19 19:09:45 by fde-alen         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "fractol.h"

/**
 * Changes thecolor for the fractal being rendered.
 *
 * This function assigns a color value to the fractal's 'color' field.
 *
 * @param[in,out] fractal A pointer to the fractal structure to be updated 
 * with the selected color.
 */
void	pick_color(t_fractal *fractal)
{
	if (fractal->color == CYAN)
		fractal->color = GOLD;
	else if (fractal->color == GOLD)
		fractal->color = TOMATO;
	else if (fractal->color == TOMATO)
		fractal->color = PINK;
	else if (fractal->color == PINK)
		fractal->color = ORANGER;
	else if (fractal->color == ORANGER)
		fractal->color = VIOLET;
	else if (fractal->color == VIOLET)
		fractal->color = TEAL;
	else if (fractal->color == TEAL)
		fractal->color = BROWN;
	else if (fractal->color == BROWN)
		fractal->color = MAGENTA;
	else if (fractal->color == MAGENTA)
		fractal->color = YELLOW;
	else
		fractal->color = CYAN;
}

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
		if (mlx_is_key_down(fractal->mlx, MLX_KEY_G))
			pick_color(fractal);
		fractal->r *= ((smoothed_factor) * 0.9);
		fractal->g *= ((smoothed_factor) * 0.9);
		fractal->b *= ((smoothed_factor) * 0.9);
	}
	return ((fractal->r << 24) | (fractal->g << 16) | (fractal->b << 8) | 255);
}
