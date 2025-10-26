# Deep-Learning-Pages


## Instalação

```bash
pip install mkdocs
```

## Criar Projeto

```bash
mkdocs new meu-projeto
cd meu-projeto
```

## Rodando Localmente

```bash
mkdocs serve -o
```

Abre automaticamente em: http://127.0.0.1:8000

## Estrutura

```
meu-projeto/
    mkdocs.yml    # Configuração
    docs/
        index.md  # Páginas em Markdown
```

## Subindo (Deploy)

### GitHub Pages

```bash
mkdocs gh-deploy
```

### Build Manual

```bash
mkdocs build
```

Arquivos gerados na pasta `site/` - faça upload para qualquer servidor.

## Comandos Essenciais

- `mkdocs serve -o` - Roda localmente e abre no navegador
- `mkdocs build` - Gera site estático
- `mkdocs gh-deploy` - Publica no GitHub Pages

---

**Pronto!** 🚀