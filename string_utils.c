/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   string_utils.c                                     :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fde-alen <fde-alen@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2023/10/07 22:57:00 by fde-alen          #+#    #+#             */
/*   Updated: 2023/10/12 18:17:56 by fde-alen         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "fractol.h"

/**
 * @brief Compares two strings
 *
 * @param str1 The first string
 * @param str2 The second string
 * @param n The number of characters to compare
 *
 * @return int 1 if the strings are equal, 0 otherwise
 */
int	ft_strncmp(const char *str1, const char *str2, size_t n)
{
	size_t	i;

	i = 0;
	if (str1 == NULL || str2 == NULL)
		return (0);
	while (i < n && str1[i] && str2[i])
	{
		if (str1[i] != str2[i])
			return (0);
		i++;
	}
	return (1);
}

/**
 * @brief Converts a string to a double
 *
 * @param str The string to convert
 *
 * @return double The converted string
 */
double	ft_atod(char *str)
{
	long	integer_part;
	double	decimal_part;
	double	decimal_factor;
	int		sign;

	integer_part = 0;
	decimal_part = 0;
	sign = 1;
	decimal_factor = 1;
	while ((*str >= 9 && *str <= 13) || *str == ' ')
		++str;
	if (*str == '-' || *str == '+')
		if (*str++ == '-')
			sign *= -1;
	while ((*str >= '0' && *str <= '9') && *str && *str != '.')
		integer_part = integer_part * 10 + (*str++ - '0');
	if (*str == '.')
		++str;
	while ((*str >= '0' && *str <= '9') && *str)
	{
		decimal_factor /= 10;
		decimal_part += (*str++ - '0') * decimal_factor;
	}
	return ((integer_part + decimal_part) * sign);
}
