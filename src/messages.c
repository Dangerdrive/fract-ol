/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   messages.c                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fde-alen <fde-alen@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2023/10/07 22:44:47 by fde-alen          #+#    #+#             */
/*   Updated: 2023/10/19 19:12:58 by fde-alen         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "fractol.h"

/**
 * Display an error message and exit the program.
 *
 * This function prints an error message to the console and exits the program
 * with a failure status.
 */
void	error(void)
{
	puts(mlx_strerror(mlx_errno));
	exit(EXIT_FAILURE);
}

/**
 * Display a guide with control instructions.
 *
 * This function prints a guide to the console, providing instructions
 * for controlling the fractal rendering program. It includes information
 * about key controls for exiting, moving, zooming, changing fractal types,
 * and randomizing the Julia set.
 */
void	guide(void)
{
	puts("Controls:\n" \
	"\tpress \033[1m\033[38;5;110mESC\033[0m to exit\n" \
	"\tpress \033[1m\033[38;5;110marrow keys\033[0m to move\n" \
	"\tuse \033[1m\033[38;5;110mmouse scroll\033[0m for zoom\n" \
	"\tpress \033[1m\033[38;5;110mTAB\033[0m to change fractal\n" \
	"\tpress \033[1m\033[38;5;110mC\033[0m to change color\n" \
	"\t keep \033[1m\033[38;5;110mG\033[0m pressed for glitchy colors\n" \
	"\nJulia Set constants:\n" \
	"\tpress \033[1m\033[38;5;110mR\033[0m to randomize \n"\
	"\tuse \033[1m\033[38;5;110mleft_shift + scroll\033[0m " \
	"to increase or decrease\n" \
	"\tuse \033[1m\033[38;5;110mleft_ctrl + mouse \033[0m"\
	"to change constants based on relative mouse position\n");
}

/**
 * Display an error message for incorrect parameters.
 *
 * This function prints an error message to the console when the program
 * is called with incorrect parameters.
 */
void	param_error(void)
{
	puts("\n\nError - incorrect params\n\n" \
"params:\t \033[1m\033[38;5;110mmandelbrot\n" \
"\t \033[1m\033[38;5;110mtricorn\n" \
"\t julia \033[0m\033[38;5;115m<real> <imaginary>\033[0m\n\n" \
"examples:\n" \
"./fractol julia \033[38;5;115m0.285 0.01\033[0m\n" \
"./fractol julia \033[38;5;115m-0.8 0.156\n");
}
