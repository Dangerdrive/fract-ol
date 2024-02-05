# Makefile Configuration
MAKEFLAGS	+=	--no-print-directory

# -I flag is used for specifying include directories to the compiler, and -L is used for specifying library directories to the linker. The -l option is used with the linker to specify the name of the library to link against

NAME	:=	fractol
CC		:=	cc
LIBMLX	:=	./MLX42
CFLAGS	:=	-Wextra -Wall -Wunreachable-code -Ofast -I./include -I$(LIBMLX)/include/MLX42
UNAME_S :=	$(shell uname -s)

ifeq ($(UNAME_S),Linux)
	LDFLAGS :=	-ldl -lglfw -pthread -lm -L$(LIBMLX)/build
endif

ifeq ($(UNAME_S),Darwin)
	ifeq ($(shell uname -m),arm64)
		LDFLAGS = -L/opt/homebrew/lib -lglfw -framework IOKit -framework Cocoa -L$(LIBMLX)/build
	else ifeq ($(shell uname -m),x86_64)
		LDFLAGS = -lglfw3 -framework IOKit -framework Cocoa -L$(LIBMLX)/build
	endif
endif

LEAKS	:=	valgrind --leak-check=full --show-leak-kinds=all --suppressions=./MLX42/suppress.sup ./fractol mandelbrot
# Update the SRCS variable to point to the new location of your source files
SRCS	:=	src/colors.c \
			src/f_julia.c \
			src/f_julia_glitch.c \
			src/f_mandelbrot.c \
			src/f_mandelbrot_glitch.c \
			src/f_tricorn.c \
			src/f_tricorn_glitch.c \
			src/fractal_init.c \
			src/hooks.c \
			src/main.c \
			src/math.c \
			src/messages.c \
			src/string_utils.c

OBJDIR	:=	obj
OBJS	:=	$(SRCS:src/%.c=$(OBJDIR)/%.o)

.PHONY:	all clean fclean re valgrind run prebuild

bonus:	all

valgrind:	all
	@$(LEAKS)

all:	prebuild $(NAME)

prebuild:
	@echo "Building MLX42..."
	@cd $(LIBMLX) && cmake -B build && cmake --build build -j4

$(NAME): $(OBJS)
	@$(CC) $(OBJS) -o $(NAME) -L$(LIBMLX) -lmlx42 $(LDFLAGS)

$(OBJDIR)/%.o: src/%.c | $(OBJDIR)
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	@mkdir -p $(OBJDIR)

clean:
	@rm -rf $(OBJDIR) $(NAME)

fclean: clean
	@rm -rf $(NAME)

re:	fclean all

run:	all
	@./fractol mandelbrot
