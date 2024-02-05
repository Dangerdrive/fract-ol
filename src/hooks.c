/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   hooks.c                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fde-alen <fde-alen@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2023/10/07 22:45:18 by fde-alen          #+#    #+#             */
/*   Updated: 2024/02/04 21:15:23 by fde-alen         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "fractol.h"

/**
 * Move the view according to the arrow keys pressed.
 *
 * @param[in,out] fractal A pointer to the fractal structure to be updated.
 */
static void	arrows_keys(t_fractal *fractal)
{
	if (mlx_is_key_down(fractal->mlx, MLX_KEY_UP))
	{
		fractal->y_shift -= 0.1 * fractal->zoom;
		update_render(fractal);
	}
	if (mlx_is_key_down(fractal->mlx, MLX_KEY_DOWN))
	{
		fractal->y_shift += 0.1 * fractal->zoom;
		update_render(fractal);
	}
	if (mlx_is_key_down(fractal->mlx, MLX_KEY_LEFT))
	{
		fractal->x_shift -= 0.1 * fractal->zoom;
		update_render(fractal);
	}
	if (mlx_is_key_down(fractal->mlx, MLX_KEY_RIGHT))
	{
		fractal->x_shift += 0.1 * fractal->zoom;
		update_render(fractal);
	}
}

/**
 * Handles keyboard input events.
 *
 * This function processes keyboard input events and updates the state
 * of the fractal 'fractal' based on the pressed keys. It supports
 * actions such as zooming, panning, color selection, and fractal type switching.
 *
 * @param[in] fractal A pointer to the fractal structure to be updated.
 */
void	keyhook(void *param)
{
	t_fractal	*fractal;

	fractal = (t_fractal *)param;
	arrows_keys(fractal);
	if (mlx_is_key_down(fractal->mlx, MLX_KEY_ESCAPE))
	{
		mlx_close_window(fractal->mlx);
	}
	else if (mlx_is_key_down(fractal->mlx, MLX_KEY_C))
	{
		pick_color(fractal);
		update_render(fractal);
	}
	else if (mlx_is_key_down(fractal->mlx, MLX_KEY_R))
	{
		randomize_julia(fractal);
		update_render(fractal);
	}
	else if (mlx_is_key_down(fractal->mlx, MLX_KEY_TAB))
	{
		select_fractal(fractal);
		update_render(fractal);
	}
	else if (mlx_is_key_down(fractal->mlx, MLX_KEY_G))
		update_render(fractal);
}

/**
 * Updates the rendering of the fractal zooming according to the scroll and
 * relative mouse position.
 *
 * @param[in] ydelta The vertical scroll delta.
 * @param[in,out] fractal A pointer to the fractal structure containing
 * rendering parameters.
 */
void	zoom(double ydelta, t_fractal *fr)
{
	double		zoom_factor;

	zoom_factor = 1.1;
	if (ydelta > 0)
	{
		fr->zoom *= 0.9;
		fr->xmin = fr->xzoom - (1.0 / zoom_factor) * (fr->xzoom - fr->xmin);
		fr->xmax = fr->xzoom + (1.0 / zoom_factor) * (fr->xmax - fr->xzoom);
		fr->ymin = fr->yzoom - (1.0 / zoom_factor) * (fr->yzoom - fr->ymin);
		fr->ymax = fr->yzoom + (1.0 / zoom_factor) * (fr->ymax - fr->yzoom);
	}
	else if (ydelta < 0)
	{
		fr->zoom *= 1.1;
		fr->xmin = fr->xzoom - zoom_factor * (fr->xzoom - fr->xmin);
		fr->xmax = fr->xzoom + zoom_factor * (fr->xmax - fr->xzoom);
		fr->ymin = fr->yzoom - zoom_factor * (fr->yzoom - fr->ymin);
		fr->ymax = fr->yzoom + zoom_factor * (fr->ymax - fr->yzoom);
	}
}

/**
 * Handles scroll wheel input events.
 *
 * This function processes scroll wheel input events and updates the render.
 * Updates the zoom of the fractal based on the scroll direction and mouse
 * position.
 * It supports increasing/decreasing the Julia set parameter 'c'.
 *
 * @param[in] xdelta The horizontal scroll delta - generally unused.
 * @param[in] ydelta The vertical scroll delta.
 * @param[in] param A pointer to the fractal structure to be updated.
 */
void	scrollhook(double xdelta, double ydelta, void *param)
{
	t_fractal	*fractal;

	(void) xdelta; // Unused
	fractal = param;
	xdelta = 0;
	mlx_get_mouse_pos(fractal->mlx, &fractal->mouse_x, &fractal->mouse_y);
	fractal->xzoom = fractal->xmin + fractal->mouse_x
		* ((fractal->xmax - fractal->xmin) / WIDTH);
	fractal->yzoom = fractal->ymin + fractal->mouse_y
		* ((fractal->ymax - fractal->ymin) / HEIGHT);
	if (fractal->id == JULIA
		&& mlx_is_key_down(fractal->mlx, MLX_KEY_LEFT_SHIFT))
	{
		fractal->c.real += (ydelta / 400) * fractal->zoom;
		fractal->c.i += (ydelta / 400) * fractal->zoom;
		update_render(fractal);
	}
	else if (ydelta != 0)
	{
		zoom(ydelta, fractal);
		update_render(fractal);
	}
}

/**
 * Handles cursor (mouse) position events.
 *
 * This function updates cursor (mouse) position on the fractal struct.
 * It supports adjusting the Julia set parameter 'c' based on
 * relative mouse position.
 *
 * @param[in] xmouse The x-coordinate of the cursor.
 * @param[in] ymouse The y-coordinate of the cursor.
 * @param[in] param A pointer to the fractal structure to be updated.
 */
void	cursorhook(double xmouse, double ymouse, void *param)
{
	t_fractal	*fractal;

	fractal = (t_fractal *)param;
	(void) xmouse; // Unused
	(void) ymouse;

	mlx_get_mouse_pos(fractal->mlx, &fractal->mouse_x,
		&fractal->mouse_y);
	if (fractal->id == JULIA
		&& mlx_is_key_down(fractal->mlx, MLX_KEY_LEFT_CONTROL))
	{
		fractal->c.real = ((fractal->mouse_x) * (fractal->xmax - fractal->xmin)
				* 0.9) / (WIDTH) + (fractal->xmin * 0.9) + fractal->x_shift;
		fractal->c.i = ((fractal->mouse_y) * (fractal->ymax - fractal->ymin)
				* 0.9) / (HEIGHT) + (fractal->ymin * 0.9) + fractal->y_shift;
		update_render(fractal);
	}
}

