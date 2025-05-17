import pygame

class Ship:
    """Класс для управления кораблем."""
    # Ссылка self и ссылка на текущий экземпляр класса AlienInvasion
    def __init__(self, ai_game):
        """Инициализирует корабль и задает начальную позицию."""
        self.screen = ai_game.screen
        self.screen_rect = ai_game.screen.get_rect()
        # Загружает изображение корабля и получает прямоугольник.
        self.image = pygame.image.load('images/ship.bmp')
        #self.original_image = pygame.image.load('images/ship.bmp')
        #self.image = pygame.transform.scale(self.original_image, (100, 100))  # ширина и высота

        self.rect = self.image.get_rect()
        # Каждый новый корабль появляется у нижнего края экрана.
        self.rect.midbottom = self.screen_rect.midbottom
        #self.rect.bottom = self.screen_rect.bottom

    def blitme(self):
        """Рисует корабль в текущей позиции."""
        self.screen.blit(self.image, self.rect)
