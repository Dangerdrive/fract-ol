/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.c                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fde-alen <fde-alen@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2023/10/07 17:49:57 by fde-alen          #+#    #+#             */
/*   Updated: 2023/10/18 12:06:25 by fde-alen         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "fractol.h"

/**
 * @brief Function to check the input and initialize the fractal.
 *
 * @param data The data struct
 */
static bool	check_input(int argc, char **argv, t_fractal *fractal)
{
	if (argc == 2 && !strncmp(argv[1], "mandelbrot", 10))
		fractal_init(fractal, MANDELBROT, 0, 0);
	else if (argc == 4 && !strncmp(argv[1], "julia", 5))
	{
		if (ft_atod(argv[2]) >= -2.0 && ft_atod(argv[2]) <= 2.0
			&& ft_atod(argv[3]) >= -2.0 && ft_atod(argv[3]) <= 2.0)
			fractal_init(fractal, JULIA, ft_atod(argv[2]), ft_atod(argv[3]));
		else
			return (false);
	}
	else if (argc == 2 && !strncmp(argv[1], "tricorn", 7))
		fractal_init(fractal, TRICORN, 0, 0);
	else
		return (false);
	return (true);
}

int	main(int argc, char **argv)
{
	t_fractal		fractal;

	if (check_input(argc, argv, &fractal))
	{
		update_render(&fractal);
		mlx_loop_hook(fractal.mlx, keyhook, &fractal);
		mlx_scroll_hook(fractal.mlx, &scrollhook, &fractal);
		mlx_cursor_hook(fractal.mlx, &cursorhook, &fractal);
		mlx_loop(fractal.mlx);
	}
	else
	{
		param_error();
		return (EXIT_FAILURE);
	}
	mlx_delete_image(fractal.mlx, fractal.img);
	mlx_terminate(fractal.mlx);
	return (EXIT_SUCCESS);
}
