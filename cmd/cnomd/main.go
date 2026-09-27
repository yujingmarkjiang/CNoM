package main

import (
	"flag"
	"log"
	"net/http"
	"os"

	"github.com/cnom/cnom/internal/api"
)

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	flag.Parse()
	key := os.Getenv("CNOM_API_KEY")
	s := api.New(key)
	log.Printf("cnomd %s", *addr)
	log.Fatal(http.ListenAndServe(*addr, s.Routes()))
}
