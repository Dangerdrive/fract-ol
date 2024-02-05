/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   math.c                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fde-alen <fde-alen@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2023/10/07 22:56:38 by fde-alen          #+#    #+#             */
/*   Updated: 2023/10/20 10:22:47 by fde-alen         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "fractol.h"

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
