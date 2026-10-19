#!/usr/bin/env bash
# o laço do start.sh também tem "app.main" na linha de comando: os dois param juntos
pkill -f -- "-m app\.main" 2>/dev/null
echo "parado"
