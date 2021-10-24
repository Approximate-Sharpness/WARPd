function Az = Axz(z)

  global Uk Vk M
  
  Az = Uk*(Vk*z) + M * z;
end