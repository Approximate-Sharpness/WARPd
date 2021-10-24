function Atz = Atxz(z)

  global Uk Vk M
  
  Atz = ((z'*Uk)*Vk + z' * M)';
end
