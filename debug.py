import pygame

def debug_text(screen, text, x=0, y=0, font_size=24, color=(0, 0, 0)):
    """Display debug text on the screen."""
    font = pygame.font.Font(None, font_size)  
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))
